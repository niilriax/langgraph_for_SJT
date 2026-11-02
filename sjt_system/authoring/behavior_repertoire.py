"""Concrete behavior repertoire and life-domain coverage for item planning.

The curated evidence families state the trait difference abstractly. The
stage-3 behavior bank lists concrete high/low behaviors copied from the Mussel
PSJT source options; stage 3b (tools/build_behavior_types.py) rewrites them
into setting-free behavior types per curated family, which are preferred when
present. Each item gets its own behaviors. Life domains spread situations
across settings and stage-4b trigger types spread them across activation
routes, so sibling items do not share one plot.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = PROJECT_ROOT / "knowledge_base" / "evidence_library"
BEHAVIOR_BANK_FILENAME = "stage3_behavior_bank.json"
BEHAVIOR_TYPES_FILENAME = "stage3b_behavior_types.json"
TRIGGER_TYPES_FILENAME = "stage4b_trigger_types.json"
MAX_MANIFESTATION_USES = 2

LIFE_DOMAINS: dict[str, tuple[str, ...]] = {
    "大学生": (
        "课堂与学业", "宿舍与室友", "社团与学生组织", "兼职、实习与求职",
        "家庭与假期", "校外出行与消费", "线上社交与网络社群", "运动、娱乐与兴趣",
        "节日、生日与庆祝", "朋友聚会与人际往来",
    ),
}
DEFAULT_LIFE_DOMAINS = (
    "学习或工作", "家庭与亲属", "朋友与人际往来", "社区与公共场所",
    "线上社交", "出行与消费", "休闲与兴趣", "节日与庆祝",
)


def life_domains_for(target_population: Any) -> list[str]:
    text = str(target_population or "")
    for key, domains in LIFE_DOMAINS.items():
        if key in text:
            return list(domains)
    return list(DEFAULT_LIFE_DOMAINS)


def domain_coverage_rule(situation_count: int, domain_count: int) -> dict[str, int]:
    return {
        "min_distinct_domains": min(domain_count, 6),
        "max_situations_per_domain": max(4, math.ceil(situation_count / 6)),
    }


def domain_coverage_problems(domains: Sequence[str], allowed: Sequence[str],
                             rule: Mapping[str, int]) -> list[str]:
    problems = []
    unknown = sorted({domain for domain in domains if domain not in allowed})
    if unknown:
        problems.append(f"domain 必须从 life_domains 中原样选择，以下不在列表中：{unknown}")
    counts = {domain: domains.count(domain) for domain in set(domains)}
    distinct = len([domain for domain in counts if domain in allowed])
    if distinct < rule["min_distinct_domains"]:
        problems.append(f"情境只覆盖了 {distinct} 个生活领域，至少需要 {rule['min_distinct_domains']} 个")
    crowded = {domain: count for domain, count in counts.items() if count > rule["max_situations_per_domain"]}
    if crowded:
        problems.append(f"以下领域情境过多（每个最多 {rule['max_situations_per_domain']} 条）：{crowded}")
    return problems


def _evidence_file(facet_id: str, filename: str) -> Path | None:
    for curated in EVIDENCE_ROOT.glob("*/stage2_evidence_library_curated.json"):
        try:
            if json.loads(curated.read_text(encoding="utf-8")).get("facet_id") == facet_id:
                path = curated.parent / filename
                return path if path.exists() else None
        except (OSError, ValueError):
            continue
    return None


def _bank_path(facet_id: str) -> Path | None:
    return _evidence_file(facet_id, BEHAVIOR_BANK_FILENAME)


def _runtime_behavior_id(payload: Mapping[str, Any], family_id: str) -> str:
    return f"{str(payload.get('neo_code') or '').upper()}_{family_id}"


def load_behavior_types(facet_id: str, behavior_evidence: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Setting-free behavior types (stage 3b) restricted to runtime families."""
    path = _evidence_file(facet_id, BEHAVIOR_TYPES_FILENAME)
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    active = {str(row.get("behavior_id")) for row in behavior_evidence}
    rows = []
    for entry in payload.get("behavior_types") or []:
        behavior_id = _runtime_behavior_id(payload, str(entry.get("family_id")))
        if behavior_id in active and entry.get("direction") in {"high", "low"} and entry.get("description"):
            rows.append({"manifestation_id": str(entry["type_id"]), "behavior_id": behavior_id,
                         "direction": entry["direction"], "description": str(entry["description"])})
    return rows


def load_manifestations(facet_id: str, behavior_evidence: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Concrete behaviors of one facet, each mapped to a runtime behavior_id.

    Setting-free behavior types are used when the facet has them. Otherwise a
    raw bank manifestation joins the behavior evidence row whose source items
    overlap its own most; manifestations with no overlap are dropped."""
    types = load_behavior_types(facet_id, behavior_evidence)
    if types:
        return types
    path = _bank_path(facet_id)
    if path is None:
        return []
    bank = json.loads(path.read_text(encoding="utf-8")).get("manifestations") or []
    rows = []
    for entry in bank:
        sources = set(entry.get("source_item_ids") or [])
        best, overlap = None, 0
        for evidence in behavior_evidence:
            shared = len(sources & set(evidence.get("source_item_ids") or []))
            if shared > overlap:
                best, overlap = evidence.get("behavior_id"), shared
        if best and entry.get("trait_direction") in {"high", "low"} and entry.get("description"):
            rows.append({"manifestation_id": str(entry["manifestation_id"]), "behavior_id": str(best),
                         "direction": entry["trait_direction"], "description": str(entry["description"])})
    return rows


def load_trigger_types(facet_id: str, behavior_evidence: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Activation routes (stage 4b) linked to at least one runtime family."""
    path = _evidence_file(facet_id, TRIGGER_TYPES_FILENAME)
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    active = {str(row.get("behavior_id")) for row in behavior_evidence}
    rows = []
    for entry in payload.get("trigger_types") or []:
        behavior_ids = [bid for bid in (_runtime_behavior_id(payload, str(fid))
                                        for fid in entry.get("linked_family_ids") or []) if bid in active]
        if behavior_ids:
            rows.append({"trigger_id": str(entry["trigger_id"]), "name": str(entry.get("name") or ""),
                         "description": str(entry.get("description") or ""), "behavior_ids": behavior_ids})
    return rows


def trigger_types_for_prompt(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{key: row[key] for key in ("trigger_id", "name", "description", "behavior_ids")} for row in rows]


def trigger_cap(item_count: int) -> int:
    """At most about a quarter of a facet's items share one trigger type."""
    return max(1, math.ceil(item_count / 4))


def trigger_coverage_rule(situation_count: int, trigger_count: int) -> dict[str, int]:
    return {"min_distinct_triggers": min(trigger_count, 4),
            "max_situations_per_trigger": max(2, trigger_cap(situation_count))}


def trigger_coverage_problems(trigger_ids: Sequence[str], triggers: Sequence[Mapping[str, Any]],
                              rule: Mapping[str, int], behavior_ids: Sequence[str] | None = None) -> list[str]:
    """Problems in the trigger_id choices of one facet's situations.

    behavior_ids, when given, is aligned with trigger_ids and checks that each
    situation's trigger is linked to its behavior evidence."""
    allowed = {row["trigger_id"]: set(row["behavior_ids"]) for row in triggers}
    problems = []
    unknown = sorted({tid for tid in trigger_ids if tid not in allowed})
    if unknown:
        problems.append(f"trigger_id 必须从 trigger_types 中选择，以下不在列表中：{unknown}")
    if behavior_ids is not None:
        mismatched = sorted({tid for tid, bid in zip(trigger_ids, behavior_ids)
                             if tid in allowed and bid not in allowed[tid]})
        if mismatched:
            problems.append(f"以下 trigger_id 与所属 behavior_id 不匹配（见 trigger_types.behavior_ids）：{mismatched}")
    counts = {tid: list(trigger_ids).count(tid) for tid in set(trigger_ids) if tid in allowed}
    if len(counts) < rule["min_distinct_triggers"]:
        problems.append(f"情境只用了 {len(counts)} 种触发方式，至少需要 {rule['min_distinct_triggers']} 种")
    crowded = {tid: count for tid, count in counts.items() if count > rule["max_situations_per_trigger"]}
    if crowded:
        problems.append(f"以下触发方式情境过多（每种最多 {rule['max_situations_per_trigger']} 条）：{crowded}")
    return problems


def manifestations_for_prompt(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, list[dict[str, str]]]]:
    grouped: dict[str, dict[str, list[dict[str, str]]]] = {}
    for row in rows:
        grouped.setdefault(row["behavior_id"], {"high": [], "low": []})[row["direction"]].append(
            {"manifestation_id": row["manifestation_id"], "description": row["description"]})
    return grouped


class ManifestationAssigner:
    """Give each item its own high/low behaviors, reusing one at most twice."""

    def __init__(self, rows: Sequence[Mapping[str, Any]]):
        self.rows = list(rows)
        self.uses: dict[str, int] = {}

    def note(self, manifestation_id: str | None) -> None:
        if manifestation_id:
            self.uses[manifestation_id] = self.uses.get(manifestation_id, 0) + 1

    def pick(self, behavior_id: str, direction: str, preferred: Sequence[str] = ()) -> dict[str, Any] | None:
        pool = [row for row in self.rows if row["direction"] == direction and row["behavior_id"] == behavior_id]
        pool = pool or [row for row in self.rows if row["direction"] == direction]
        if not pool:
            return None
        ranked = sorted(pool, key=lambda row: (
            self.uses.get(row["manifestation_id"], 0) >= MAX_MANIFESTATION_USES,
            row["manifestation_id"] not in preferred,
            self.uses.get(row["manifestation_id"], 0),
            row["manifestation_id"],
        ))
        chosen = ranked[0]
        self.note(chosen["manifestation_id"])
        return dict(chosen)


def rebalance_candidate_references(rows: Sequence[Mapping[str, Any]], expansions: Sequence[Any]) -> list[dict[str, Any]]:
    """Spread each facet's selected situations across triggers, domains and mechanisms.

    Only mechanisms the blueprint proposal already selected for a behavior are
    eligible (they passed its construct check); within them, situations are
    re-picked greedily to keep any trigger type within about a quarter of the
    facet, avoid repeating a mechanism more than twice, and prefer unused
    trigger types and life domains."""
    situations: dict[tuple[str, str], list[tuple[str, str, str, str]]] = {}
    for expansion in expansions:
        for behavior in expansion.behavior_expansions:
            for mechanism in behavior.mechanisms:
                for situation in mechanism.situations:
                    situations.setdefault((expansion.facet_id, behavior.behavior_id), []).append(
                        (mechanism.mechanism_id, situation.situation_id, situation.domain,
                         str(getattr(situation, "trigger_id", "") or "")))
    result = [dict(row) for row in rows]
    facets = {str(row["facet_id"]) for row in result}
    for facet_id in facets:
        facet_rows = [row for row in result if str(row["facet_id"]) == facet_id]
        cap = trigger_cap(len(facet_rows))
        vetted: dict[str, set[str]] = {}
        for row in facet_rows:
            for ref in row["candidate_references"]:
                vetted.setdefault(str(row["behavior_id"]), set()).add(str(ref["mechanism_id"]))
        used_situations: set[str] = set()
        mechanism_uses: dict[str, int] = {}
        domain_uses: dict[str, int] = {}
        trigger_uses: dict[str, int] = {}
        for row in facet_rows:
            pool = [entry for entry in situations.get((facet_id, str(row["behavior_id"])), [])
                    if entry[0] in vetted.get(str(row["behavior_id"]), set()) and entry[1] not in used_situations]
            if not pool:
                for ref in row["candidate_references"]:
                    used_situations.add(str(ref["situation_id"]))
                continue
            mechanism_id, situation_id, domain, trigger_id = min(pool, key=lambda entry: (
                bool(entry[3]) and trigger_uses.get(entry[3], 0) >= cap,
                mechanism_uses.get(entry[0], 0) >= 2,
                trigger_uses.get(entry[3], 0) if entry[3] else 0,
                domain_uses.get(entry[2], 0), mechanism_uses.get(entry[0], 0), entry[1]))
            row["candidate_references"] = [{"mechanism_id": mechanism_id, "situation_id": situation_id}]
            used_situations.add(situation_id)
            mechanism_uses[mechanism_id] = mechanism_uses.get(mechanism_id, 0) + 1
            domain_uses[domain] = domain_uses.get(domain, 0) + 1
            if trigger_id:
                trigger_uses[trigger_id] = trigger_uses.get(trigger_id, 0) + 1
    return result


def situation_triggers(expansions: Sequence[Any]) -> dict[str, str]:
    """situation_id -> trigger_id for every expanded situation that has one."""
    return {situation.situation_id: str(getattr(situation, "trigger_id", "") or "")
            for expansion in expansions for behavior in expansion.behavior_expansions
            for mechanism in behavior.mechanisms for situation in mechanism.situations
            if getattr(situation, "trigger_id", "")}


def trigger_brief(facet_id: str, behavior_evidence: Sequence[Mapping[str, Any]], trigger_id: str | None) -> dict[str, str] | None:
    """Name and description of one trigger type, for skeleton and writer prompts."""
    if not trigger_id:
        return None
    row = next((row for row in load_trigger_types(facet_id, behavior_evidence) if row["trigger_id"] == trigger_id), None)
    return {"trigger_id": row["trigger_id"], "name": row["name"], "description": row["description"]} if row else None
