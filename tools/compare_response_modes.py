"""Compare whole-form batch answering with independent single-item answering.

Reuses items and virtual respondents from an existing run checkpoint, administers
the same items to the same respondents twice in each mode, and scores responses
against each respondent's assigned facet scores (ground truth, no IPIP layer).

Usage:
    python -X utf8 tools/compare_response_modes.py --source-run <run_id> \
        [--items-per-facet 8] [--respondents 60] [--out-dir DIR]

Re-running with the same --out-dir resumes and skips completed calls.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from sjt_system.agent.client import get_model  # noqa: E402
from sjt_system.evaluation.simulation import (  # noqa: E402
    balanced_option_order,
    build_persona_prompt,
    build_sjt_batch_messages,
    build_sjt_messages,
)

MODES = ("batch", "single")
ADMINISTRATIONS = (1, 2)
MAX_ATTEMPTS = 5
CONCURRENCY = 48
SEED = 20261015
_JSON = re.compile(r"\{.*\}", re.S)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_json(text: str) -> dict:
    match = _JSON.search(text or "")
    if not match:
        raise ValueError("no JSON object in output")
    return json.loads(match.group(0))


def load_inputs(source_run: str, items_per_facet: int, respondent_count: int):
    path = PROJECT_ROOT / "outputs" / "run_checkpoints" / f"{source_run}.json"
    state = json.loads(path.read_text(encoding="utf-8"))["state"]
    config = state["virtual_sample_config"]
    facets = list(state["test_specification"]["facet_item_counts"])
    items = []
    for facet in facets:
        chosen = [item for item in state["item_pool"] if item["target_dimension_id"] == facet][:items_per_facet]
        if len(chosen) < items_per_facet:
            raise ValueError(f"{facet} has only {len(chosen)} items")
        items.extend(chosen)
    respondents = state["virtual_respondents"][:respondent_count]
    personas = {
        row["respondent_id"]: build_persona_prompt(
            row, score_specs=config["score_specs"],
            demographics_snapshot=config.get("demographics_snapshot"))
        for row in respondents
    }
    return state, config, facets, items, respondents, personas


def option_order(item, respondent_index, administration):
    return balanced_option_order(
        [row["option_id"] for row in item["response_options"]],
        respondent_index=respondent_index, seed=SEED + administration, item_id=item["item_id"])


def score_of(item, order, letter):
    original = order[ord(letter) - ord("A")]
    return int(item["scoring_key"][original])


class Runner:
    def __init__(self, out_dir: Path, model, items, respondents, personas):
        self.out_dir = out_dir
        self.model = model
        self.items = items
        self.respondents = respondents
        self.personas = personas
        self.path = out_dir / "responses.jsonl"
        self.done = set()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.done.add(row["call_key"])
        self.lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(CONCURRENCY)
        self.progress = 0
        self.total = 0

    async def write(self, row):
        async with self.lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.progress += 1
            if self.progress % 50 == 0 or self.progress == self.total:
                print(f"[progress] {self.progress}/{self.total} calls", flush=True)

    async def invoke(self, messages):
        async with self.semaphore:
            message = await self.model.ainvoke(messages)
        usage = getattr(message, "usage_metadata", None) or {}
        return message.content, int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)

    async def run_batch(self, index, respondent, administration):
        key = f"batch|{administration}|{respondent['respondent_id']}"
        if key in self.done:
            return
        order_rng = random.Random(f"{SEED}|{administration}|{respondent['respondent_id']}")
        items = list(self.items)
        order_rng.shuffle(items)
        orders = {item["item_id"]: option_order(item, index, administration) for item in items}
        persona = self.personas[respondent["respondent_id"]]
        messages = build_sjt_batch_messages([(item, persona, orders[item["item_id"]]) for item in items])
        errors, tokens = [], [0, 0]
        answers = None
        for _attempt in range(MAX_ATTEMPTS):
            try:
                text, prompt_tokens, completion_tokens = await self.invoke(messages)
                tokens[0] += prompt_tokens
                tokens[1] += completion_tokens
                rows = parse_json(text).get("responses")
                parsed = {str(row.get("item_id")): row.get("selected_option_id") for row in rows or []
                          if isinstance(row, dict)}
                if set(parsed) != set(orders) or any(
                        value not in {"A", "B", "C", "D"} for value in parsed.values()):
                    raise ValueError(f"incomplete batch: {len(parsed)}/{len(orders)} items")
                answers = parsed
                break
            except Exception as exc:  # noqa: BLE001 - recorded, bounded retries
                errors.append(f"{type(exc).__name__}: {exc}"[:300])
        responses = []
        if answers:
            for position, item in enumerate(items, 1):
                letter = answers[item["item_id"]]
                responses.append({"item_id": item["item_id"], "position": position, "letter": letter,
                                  "score": score_of(item, orders[item["item_id"]], letter)})
        await self.write({"call_key": key, "mode": "batch", "administration": administration,
                          "respondent_id": respondent["respondent_id"], "responses": responses,
                          "attempts": len(errors) + (1 if answers else 0), "errors": errors,
                          "prompt_tokens": tokens[0], "completion_tokens": tokens[1], "recorded_at": now()})

    async def run_single(self, index, respondent, item, administration):
        key = f"single|{administration}|{respondent['respondent_id']}|{item['item_id']}"
        if key in self.done:
            return
        order = option_order(item, index, administration)
        messages = build_sjt_messages(self.personas[respondent["respondent_id"]], item,
                                      display_option_order=order)
        errors, tokens, letter = [], [0, 0], None
        for _attempt in range(MAX_ATTEMPTS):
            try:
                text, prompt_tokens, completion_tokens = await self.invoke(messages)
                tokens[0] += prompt_tokens
                tokens[1] += completion_tokens
                value = parse_json(text).get("selected_option_id")
                if value not in {"A", "B", "C", "D"}:
                    raise ValueError(f"invalid option {value!r}")
                letter = value
                break
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{type(exc).__name__}: {exc}"[:300])
        responses = ([{"item_id": item["item_id"], "position": None, "letter": letter,
                       "score": score_of(item, order, letter)}] if letter else [])
        await self.write({"call_key": key, "mode": "single", "administration": administration,
                          "respondent_id": respondent["respondent_id"], "responses": responses,
                          "attempts": len(errors) + (1 if letter else 0), "errors": errors,
                          "prompt_tokens": tokens[0], "completion_tokens": tokens[1], "recorded_at": now()})

    async def run(self):
        jobs = []
        for administration in ADMINISTRATIONS:
            for index, respondent in enumerate(self.respondents):
                jobs.append(self.run_batch(index, respondent, administration))
                for item in self.items:
                    jobs.append(self.run_single(index, respondent, item, administration))
        self.total = len(jobs)
        self.progress = len(self.done)
        print(f"[start] {self.total} calls, {len(self.done)} already done", flush=True)
        await asyncio.gather(*jobs)


# ---------------------------------------------------------------- analysis

def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 5 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return None
    return float(spearmanr(x[mask], y[mask]).statistic)


def cronbach_alpha(matrix):
    matrix = matrix[~np.isnan(matrix).any(axis=1)]
    k = matrix.shape[1]
    if matrix.shape[0] < 5 or k < 2:
        return None
    total_var = matrix.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return None
    return float(k / (k - 1) * (1 - matrix.var(axis=0, ddof=1).sum() / total_var))


def icc_a1(first, second):
    """Two-way random, absolute agreement, single measure ICC(A,1)."""
    data = np.column_stack([first, second])
    data = data[~np.isnan(data).any(axis=1)]
    n, k = data.shape
    if n < 5:
        return None
    grand = data.mean()
    ms_rows = k * ((data.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_cols = n * ((data.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    residual = data - data.mean(axis=1, keepdims=True) - data.mean(axis=0, keepdims=True) + grand
    ms_err = (residual ** 2).sum() / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n
    return float((ms_rows - ms_err) / denominator) if denominator else None


def fmt(value):
    return "—" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.3f}"


def analyse(out_dir: Path, facets, items, respondents):
    rows = [json.loads(line) for line in (out_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines()]
    ids = [row["respondent_id"] for row in respondents]
    item_ids = [item["item_id"] for item in items]
    facet_of = {item["item_id"]: item["target_dimension_id"] for item in items}
    scores = {row["respondent_id"]: row["score_values"] for row in respondents}
    matrices = {(mode, adm): np.full((len(ids), len(item_ids)), np.nan) for mode in MODES for adm in ADMINISTRATIONS}
    positions = []
    usage = {mode: {"calls": 0, "attempts": 0, "failed": 0, "prompt_tokens": 0, "completion_tokens": 0}
             for mode in MODES}
    for row in rows:
        stats = usage[row["mode"]]
        stats["calls"] += 1
        stats["attempts"] += row["attempts"]
        stats["failed"] += 0 if row["responses"] else 1
        stats["prompt_tokens"] += row["prompt_tokens"]
        stats["completion_tokens"] += row["completion_tokens"]
        r = ids.index(row["respondent_id"])
        for response in row["responses"]:
            c = item_ids.index(response["item_id"])
            matrices[(row["mode"], row["administration"])][r, c] = response["score"]
            if row["mode"] == "batch" and row["administration"] == 1:
                positions.append((response["position"], response["item_id"], row["respondent_id"], response["score"]))

    report = {"facets": {}, "items": {}, "usage": usage, "agreement": {}, "position": {}}
    for facet in facets:
        columns = [i for i, item_id in enumerate(item_ids) if facet_of[item_id] == facet]
        others = [f for f in facets if f != facet]
        target = np.array([scores[i][facet] for i in ids])
        report["facets"][facet] = {}
        for mode in MODES:
            m1, m2 = matrices[(mode, 1)][:, columns], matrices[(mode, 2)][:, columns]
            total1 = np.where(np.isnan(m1).any(axis=1), np.nan, m1.sum(axis=1))
            total2 = np.where(np.isnan(m2).any(axis=1), np.nan, m2.sum(axis=1))
            rho_t = spearman(total1, target)
            rho_o = max((abs(spearman(total1, [scores[i][o] for i in ids]) or 0) for o in others), default=None)
            complete = m1[~np.isnan(m1).any(axis=1)]
            corr = np.corrcoef(complete.T) if complete.shape[0] > 5 else None
            mean_r = (float(np.nanmean(corr[np.triu_indices(len(columns), 1)])) if corr is not None else None)
            report["facets"][facet][mode] = {
                "alpha": cronbach_alpha(m1), "mean_inter_item_r": mean_r,
                "retest_icc": icc_a1(total1, total2), "target_rho": rho_t,
                "max_abs_non_target_rho": rho_o,
                "delta": (rho_t - rho_o) if rho_t is not None and rho_o is not None else None,
            }
        for column in columns:
            item_id = item_ids[column]
            report["items"][item_id] = {"facet": facet}
            for mode in MODES:
                m1 = matrices[(mode, 1)][:, columns]
                own = matrices[(mode, 1)][:, column]
                rest = np.nansum(m1, axis=1) - np.nan_to_num(own)
                observed = ~np.isnan(own)
                report["items"][item_id][mode] = {
                    "citc": (float(np.corrcoef(own[observed], rest[observed])[0, 1])
                             if np.std(own[observed]) > 0 and np.std(rest[observed]) > 0 else None),
                    "target_rho": spearman(own, target),
                    "mean_score": float(np.nanmean(own)),
                    "retest_exact_agreement": float(np.nanmean(
                        matrices[(mode, 1)][:, column] == matrices[(mode, 2)][:, column])),
                }
    a, b = matrices[("batch", 1)], matrices[("single", 1)]
    valid = ~(np.isnan(a) | np.isnan(b))
    report["agreement"]["batch_vs_single_same_option_rate"] = float((a[valid] == b[valid]).mean())
    half = len(item_ids) // 2
    for label, keep in (("first_half", lambda p: p <= half), ("second_half", lambda p: p > half)):
        chosen = [(item_id, rid, score) for position, item_id, rid, score in positions if keep(position)]
        report["position"][label] = spearman(
            [score for _, _, score in chosen], [scores[rid][facet_of[item_id]] for item_id, rid, _ in chosen])
    (out_dir / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 整卷作答 vs 单题作答对照", "",
             f"- 题目：{len(items)} 道（每 facet {len(items) // len(facets)} 道）；被试：{len(ids)} 名；每种方式施测 2 次。",
             "- 指标以被试的设定 facet 分数为真值；ICC 为两次施测 facet 总分的 ICC(A,1)。", "",
             "## Facet 整卷指标", "",
             "| facet | 方式 | α | 平均题间r | 重测ICC | 目标ρ | 最大非目标|ρ| | Δ |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for facet, modes in report["facets"].items():
        for mode, row in modes.items():
            lines.append(f"| {facet} | {mode} | {fmt(row['alpha'])} | {fmt(row['mean_inter_item_r'])} | "
                         f"{fmt(row['retest_icc'])} | {fmt(row['target_rho'])} | "
                         f"{fmt(row['max_abs_non_target_rho'])} | {fmt(row['delta'])} |")
    lines += ["", "## 单题指标（第1次施测）", "",
              "| 题目 | facet | CITC 整卷/单题 | 目标ρ 整卷/单题 | 均分 整卷/单题 | 重测一致率 整卷/单题 |",
              "|---|---|---|---|---|---|"]
    for item_id, row in report["items"].items():
        b_, s_ = row["batch"], row["single"]
        lines.append(f"| {item_id[-14:]} | {row['facet'].split('_', 1)[1]} | {fmt(b_['citc'])} / {fmt(s_['citc'])} | "
                     f"{fmt(b_['target_rho'])} / {fmt(s_['target_rho'])} | {fmt(b_['mean_score'])} / "
                     f"{fmt(s_['mean_score'])} | {fmt(b_['retest_exact_agreement'])} / "
                     f"{fmt(s_['retest_exact_agreement'])} |")
    lines += ["", "## 其他", "",
              f"- 同一被试同一题，两种方式选了同一选项的比例：{fmt(report['agreement']['batch_vs_single_same_option_rate'])}",
              f"- 整卷模式位置效应（题目分与设定分数的合并ρ）：前半 {fmt(report['position']['first_half'])}，"
              f"后半 {fmt(report['position']['second_half'])}",
              "", "## 调用", "", "| 方式 | 调用 | 总尝试 | 最终失败 | 输入token | 输出token |", "|---|---:|---:|---:|---:|---:|"]
    for mode, stats in usage.items():
        lines.append(f"| {mode} | {stats['calls']} | {stats['attempts']} | {stats['failed']} | "
                     f"{stats['prompt_tokens']} | {stats['completion_tokens']} |")
    (out_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--items-per-facet", type=int, default=8)
    parser.add_argument("--respondents", type=int, default=60)
    parser.add_argument("--model-id", default=os.getenv("VIRTUAL_RESPONDENT_MODEL_ID") or "deepseek-v4.1-flash")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--analyse-only", action="store_true")
    args = parser.parse_args()
    state, config, facets, items, respondents, personas = load_inputs(
        args.source_run, args.items_per_facet, args.respondents)
    out_dir = args.out_dir or (PROJECT_ROOT / "outputs" / "response_mode_comparison"
                               / datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"source_run": args.source_run, "model_id": args.model_id,
            "temperature": config["response_temperature"], "thinking": "disabled",
            "items": [item["item_id"] for item in items], "respondents": [r["respondent_id"] for r in respondents],
            "seed": SEED, "max_attempts": MAX_ATTEMPTS}
    meta_path = out_dir / "meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8")) != meta:
        raise ValueError("out-dir belongs to a different configuration")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {out_dir}", flush=True)
    if not args.analyse_only:
        model = get_model(args.model_id, temperature=config["response_temperature"], thinking_type="disabled")
        asyncio.run(Runner(out_dir, model, items, respondents, personas).run())
    analyse(out_dir, facets, items, respondents)


if __name__ == "__main__":
    main()
