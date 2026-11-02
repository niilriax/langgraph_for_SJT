"""Export a respondent-facing pilot form and a separate scoring key.

The respondent form interleaves facets, shows no facet labels and no scoring
section; the key (CSV) maps each displayed item and option to its facet and
score for the researchers.

Usage:
    python -X utf8 tools/export_pilot_form.py --run-id <run_id> [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import csv
import glob
import html
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PAGE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>情境判断问卷</title>
<style>
body {{ font-family: 'Noto Sans SC','Microsoft YaHei',sans-serif; font-size: 12pt; line-height: 1.7;
       color: #222; max-width: 760px; margin: 32px auto; padding: 0 16px; }}
h1 {{ font-size: 20pt; text-align: center; }}
.notice {{ background: #f5f5f5; padding: 12px 16px; border-radius: 6px; margin-bottom: 24px; }}
.item {{ margin: 22px 0; page-break-inside: avoid; }}
.q {{ margin-bottom: 6px; }}
.opt {{ margin: 3px 0 3px 1.5em; }}
</style></head><body>
<h1>情境判断问卷</h1>
<div class="notice">请阅读每个情境，从四个选项中选出<strong>你最可能采取</strong>的做法。
选项没有对错之分，请按照你平时的真实习惯作答。每题只选一项。</div>
{items}
</body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    final_path = glob.glob(str(PROJECT_ROOT / "outputs" / "final_reports" / args.run_id / "*" / "final_test.json"))
    if len(final_path) != 1:
        raise SystemExit(f"expected one final_test.json for {args.run_id}, found {len(final_path)}")
    test = json.loads(Path(final_path[0]).read_text(encoding="utf-8"))
    form = {row["public_item_id"]: row for row in test["respondent_form"]["items"]}
    keys = {row["public_item_id"]: row for row in test["scoring_key"]["items"]}

    by_facet: dict[str, list[str]] = {}
    for public_id in form:
        by_facet.setdefault(str(keys[public_id]["target_dimension_id"]), []).append(public_id)
    order, queues = [], [list(ids) for ids in by_facet.values()]
    while any(queues):  # interleave facets so neighbouring items differ
        for queue in queues:
            if queue:
                order.append(queue.pop(0))

    out_dir = args.out_dir or (PROJECT_ROOT / "outputs" / "pilot_forms" / args.run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    blocks, rows = [], []
    for number, public_id in enumerate(order, 1):
        item, key = form[public_id], keys[public_id]
        options = "".join(f'<p class="opt">{html.escape(o["option_id"])}. {html.escape(o["text"])}</p>'
                          for o in item["response_options"])
        blocks.append(f'<div class="item"><p class="q"><strong>{number}.</strong> '
                      f'{html.escape(item["scenario"])}</p><p class="q">{html.escape(item["response_instruction"])}</p>'
                      f'{options}</div>')
        scores = key["scoring_key"]
        rows.append({"display_number": number, "public_item_id": public_id,
                     "facet": key["target_dimension_id"],
                     **{f"score_{oid}": scores.get(oid) for oid in "ABCD"}})
    (out_dir / "respondent_form.html").write_text(PAGE.format(items="\n".join(blocks)), encoding="utf-8")
    with (out_dir / "scoring_key.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "README.txt").write_text(
        f"来源运行：{args.run_id}\n"
        "respondent_form.html：给被试的作答卷（facet 交替排列，不显示维度名称和计分说明）。\n"
        "scoring_key.csv：研究者用计分键，display_number 为作答卷题号，score_A..D 为各选项得分（1–4），"
        "facet 为所属分面；facet 得分 = 该 facet 各题得分之和或均值。\n"
        "注意：题目指标来自开发期虚拟被试，尚无真人常模与效度证据。\n", encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
