"""Offline stage 3b/4b: de-situated behavior types and trigger types.

Stage 3 (behavior bank) lists behaviors copied from the Mussel PSJT options,
so each behavior carries its source scene (wrong cinema hall, elevator, ...).
Stage 3b rewrites them into setting-free behavior types grouped by the curated
evidence family; every source manifestation must land in exactly one type, so
no new evidence is created.

Stage 4 (cue bank) is grouped by surface events. Stage 4b groups the cues into
trigger types (how the situation activates the facet) and may add a few types
derived from the facet definition, marked origin="definition" for review.

Usage:
    venv/bin/python -X utf8 tools/build_behavior_types.py [--facets self_consciousness,gregariousness] [--force]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sjt_system.authoring.behavior_repertoire import (  # noqa: E402
    BEHAVIOR_TYPES_FILENAME, EVIDENCE_ROOT, TRIGGER_TYPES_FILENAME,
)
from sjt_system.runtime.io import write_json_atomic  # noqa: E402

MAX_ATTEMPTS = 3


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BehaviorType(Strict):
    type_id: str = Field(description="BT01, BT02, ...")
    family_id: str = Field(description="curated evidence family id, e.g. NEF01")
    direction: Literal["high", "low"]
    description: str = Field(description="setting-free observable action type, one short clause")
    psychological_function: str = Field(description="what the action does for the person, one short clause")
    source_manifestation_ids: list[str] = Field(min_length=1)


class BehaviorTypeBank(Strict):
    behavior_types: list[BehaviorType] = Field(min_length=2)


class TriggerType(Strict):
    trigger_id: str = Field(description="TT01, TT02, ...")
    name: str = Field(description="short name, at most 12 Chinese characters")
    description: str = Field(description="setting-free description of how the situation activates the facet")
    origin: Literal["source", "definition"]
    linked_family_ids: list[str] = Field(min_length=1)
    source_cue_ids: list[str] = Field(default_factory=list)


class TriggerTypeBank(Strict):
    trigger_types: list[TriggerType] = Field(min_length=3, max_length=8)


BEHAVIOR_TYPES_PROMPT = """\
You turn a PSJT behavior bank into setting-free behavior types for one Big Five
facet. The bank's behaviors were copied from source item options, so each one
carries its source scene (a cinema, an elevator, a bank advisor, a phone).

INPUT
- facet: definition and poles
- families: curated evidence families (family_id, name, definition, high/low evidence)
- manifestations: source behaviors with manifestation_id, direction, description,
  and candidate_family_ids (families sharing source items)

TASK
Group the manifestations into behavior types. A behavior type:
- names the observable action type and its direction, without any place,
  object, role, or event from the source scene. It must fit many different
  everyday situations. Example: "留在错误影厅不换厅，因为让半排人站起来不舒服"
  becomes "为了不引人注意，宁可忍受不便也不去纠正".
- belongs to exactly one family_id taken from candidate_family_ids of its
  members and keeps their direction.
- merges manifestations that are the same action type in different scenes;
  keeps different action types (leave, hide, stay silent, joke, explain,
  apologise, ask for help ...) apart.
- never uses words that name the trait or its poles.

Every manifestation_id must appear in exactly one type. Do not invent
behaviors that no manifestation supports. Write in Chinese.
Return only the structured object.
"""

TRIGGER_TYPES_PROMPT = """\
You define trigger types for one Big Five facet: the different ways an
everyday situation can activate the facet. They are used to keep generated
situations from all using the same plot.

INPUT
- facet: definition and poles
- families: curated evidence families with their common_trait_activating_cues
- cues: source cue bank entries (cue_id, description, category)

TASK
1. Group the source cues by HOW the facet is activated (for example: attention
   already on the self after a slip; an upcoming chance to be evaluated; an
   ambiguous reaction from others; a companion's behavior reflecting on the
   self), not by setting or event surface. These have origin "source" and list
   their source_cue_ids.
2. Read the facet definition. If an important way of activating the facet is
   missing from the source cues, add at most 3 trigger types with origin
   "definition" and empty source_cue_ids.
Each trigger type is setting-free, links to the family_ids it can activate,
and differs from the others in its activation route. Return 4 to 8 types in
Chinese. Return only the structured object.
"""


def _facet_dirs(names: list[str] | None) -> list[Path]:
    dirs = sorted(path.parent for path in EVIDENCE_ROOT.glob("*/stage2_evidence_library_curated.json"))
    return [path for path in dirs if not names or path.name in names]


def _families(curated: dict) -> list[dict]:
    return [{key: family.get(key) for key in (
        "family_id", "family_name", "definition", "high_trait_evidence", "low_trait_evidence",
        "status", "common_trait_activating_cues")} for family in curated["evidence_library"]]


def _candidate_families(manifestation: dict, curated: dict) -> list[str]:
    sources = set(manifestation.get("source_item_ids") or [])
    scored = []
    for family in curated["evidence_library"]:
        shared = len(sources & set(family.get("supporting_item_ids") or []))
        if shared:
            scored.append((shared, family["family_id"]))
    return [family_id for _, family_id in sorted(scored, reverse=True)]


async def _invoke(agent, payload: dict, label: str, output_type: type[BaseModel], validate) -> BaseModel:
    from sjt_system.agent.retry import ainvoke_model_with_schema_repair
    feedback = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        data = {**payload, **({"revision_instruction": feedback} if feedback else {})}
        raw = await ainvoke_model_with_schema_repair(agent, {"input_data": data}, job_label=f"{label}#{attempt}")
        result = raw if isinstance(raw, output_type) else output_type.model_validate(raw)
        problems = validate(result)
        if not problems:
            return result
        feedback = "上一版有以下问题，请修正：" + "；".join(problems)
        print(f"  {label} attempt {attempt}: {problems}", flush=True)
    raise ValueError(f"{label} failed validation: {problems}")


async def build_facet(folder: Path, agents: dict, force: bool) -> None:
    curated = json.loads((folder / "stage2_evidence_library_curated.json").read_text(encoding="utf-8"))
    facet = {key: curated.get(key) for key in ("facet_id", "neo_code", "definition", "high_pole", "low_pole")}
    families = _families(curated)
    family_ids = {row["family_id"] for row in families}

    types_path = folder / BEHAVIOR_TYPES_FILENAME
    if force or not types_path.exists():
        bank = json.loads((folder / "stage3_behavior_bank.json").read_text(encoding="utf-8"))["manifestations"]
        manifestations = []
        for row in bank:
            candidates = _candidate_families(row, curated)
            if candidates:
                manifestations.append({"manifestation_id": row["manifestation_id"], "direction": row["trait_direction"],
                                       "description": row["description"], "candidate_family_ids": candidates})
        by_id = {row["manifestation_id"]: row for row in manifestations}

        def validate_types(result: BehaviorTypeBank) -> list[str]:
            problems = []
            seen: dict[str, str] = {}
            for row in result.behavior_types:
                for mid in row.source_manifestation_ids:
                    if mid not in by_id:
                        problems.append(f"{row.type_id} 引用了不存在的 {mid}")
                        continue
                    if mid in seen:
                        problems.append(f"{mid} 同时出现在 {seen[mid]} 和 {row.type_id}")
                    seen[mid] = row.type_id
                    if by_id[mid]["direction"] != row.direction:
                        problems.append(f"{row.type_id} 方向与 {mid} 不一致")
                    if row.family_id not in by_id[mid]["candidate_family_ids"]:
                        problems.append(f"{row.type_id} 的 family_id 不在 {mid} 的 candidate_family_ids 中")
            missing = sorted(set(by_id) - set(seen))
            if missing:
                problems.append(f"以下 manifestation 没有归入任何类型：{missing}")
            return problems

        result = await _invoke(agents["types"], {"facet": facet, "families": families, "manifestations": manifestations},
                               f"stage3b/{folder.name}", BehaviorTypeBank, validate_types)
        write_json_atomic(types_path, {"facet_id": facet["facet_id"], "neo_code": facet["neo_code"],
                                       "source": "stage3_behavior_bank.json",
                                       "behavior_types": [row.model_dump() for row in result.behavior_types]})
        print(f"  stage3b {folder.name}: {len(manifestations)} manifestations -> {len(result.behavior_types)} types", flush=True)

    triggers_path = folder / TRIGGER_TYPES_FILENAME
    if force or not triggers_path.exists():
        cues = json.loads((folder / "stage4_cue_bank.json").read_text(encoding="utf-8"))["cues"]
        cue_ids = {row["cue_id"] for row in cues}

        def validate_triggers(result: TriggerTypeBank) -> list[str]:
            problems = []
            for row in result.trigger_types:
                unknown = set(row.linked_family_ids) - family_ids
                if unknown:
                    problems.append(f"{row.trigger_id} 关联了不存在的证据族 {sorted(unknown)}")
                if row.origin == "source" and not row.source_cue_ids:
                    problems.append(f"{row.trigger_id} 来源为 source 但没有 source_cue_ids")
                if set(row.source_cue_ids) - cue_ids:
                    problems.append(f"{row.trigger_id} 引用了不存在的线索")
            if sum(row.origin == "definition" for row in result.trigger_types) > 3:
                problems.append("依定义补充的触发方式最多 3 种")
            return problems

        result = await _invoke(agents["triggers"], {
            "facet": facet, "families": families,
            "cues": [{key: row.get(key) for key in ("cue_id", "description", "cue_category")} for row in cues],
        }, f"stage4b/{folder.name}", TriggerTypeBank, validate_triggers)
        write_json_atomic(triggers_path, {"facet_id": facet["facet_id"], "neo_code": facet["neo_code"],
                                          "source": "stage4_cue_bank.json + facet definition",
                                          "trigger_types": [row.model_dump() for row in result.trigger_types]})
        print(f"  stage4b {folder.name}: {len(result.trigger_types)} trigger types", flush=True)


async def main() -> int:
    from dotenv import load_dotenv
    from sjt_system.agent.agent_factory import create_agent
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facets", help="comma-separated evidence folder names")
    parser.add_argument("--force", action="store_true", help="rebuild existing files")
    args = parser.parse_args()
    names = [name.strip() for name in args.facets.split(",")] if args.facets else None
    agents = {"types": create_agent(BEHAVIOR_TYPES_PROMPT, BehaviorTypeBank, temperature=0.1),
              "triggers": create_agent(TRIGGER_TYPES_PROMPT, TriggerTypeBank, temperature=0.1)}
    await asyncio.gather(*(build_facet(folder, agents, args.force) for folder in _facet_dirs(names)))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
