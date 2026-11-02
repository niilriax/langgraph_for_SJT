"""Population-specific behavior expansion and two-way blueprint proposals."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
import os
from typing import Any

from pydantic import Field

from sjt_system.agent.retry import ainvoke_model_with_schema_repair
from sjt_system.knowledge.behavior_evidence import StrictModel
from sjt_system.prompt.blueprint_prompt import BLUEPRINT_PROMPT
from sjt_system.prompt.situation_expansion_prompt import EXPANSION_PROMPT
from sjt_system.runtime.io import write_json_atomic


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_KNOWLEDGE_ROOT = PROJECT_ROOT / "outputs" / "run_knowledge"

_MAX_VALIDATION_RETRIES = 2
BLUEPRINT_SEMANTIC_RETRY_ATTEMPTS = 2
INCREMENTAL_CANDIDATES_PER_CELL = 1


class SituationDraft(StrictModel):
    domain: str = Field(min_length=1, max_length=100)
    actor_relation: str = Field(min_length=1, max_length=160)
    event_class: str = Field(min_length=1, max_length=200)
    # Supplied concrete behaviors (high and low) that are realistic options here.
    manifestation_ids: list[str] = Field(default_factory=list, max_length=8)
    # Supplied trigger type (activation route) this situation realizes.
    trigger_id: str = Field(default="", max_length=40)


class MechanismDraft(StrictModel):
    activation_mechanism: str = Field(min_length=1, max_length=400)
    situations: list[SituationDraft] = Field(min_length=1)


class BehaviorExpansionDraft(StrictModel):
    behavior_id: str = Field(min_length=1)
    mechanisms: list[MechanismDraft] = Field(min_length=1, max_length=4)


class FacetExpansionAgentOutput(StrictModel):
    behavior_expansions: list[BehaviorExpansionDraft] = Field(min_length=1)


class SituationRecord(SituationDraft):
    situation_id: str = Field(min_length=1)


class MechanismRecord(StrictModel):
    mechanism_id: str = Field(min_length=1)
    activation_mechanism: str = Field(min_length=1, max_length=400)
    situations: list[SituationRecord] = Field(min_length=1)


class BehaviorExpansionRecord(StrictModel):
    behavior_id: str = Field(min_length=1)
    mechanisms: list[MechanismRecord] = Field(min_length=1, max_length=4)


class FacetExpansion(StrictModel):
    facet_id: str = Field(min_length=1)
    target_population: str = Field(min_length=1)
    behavior_expansions: list[BehaviorExpansionRecord] = Field(min_length=1)


class BlueprintCandidateReference(StrictModel):
    mechanism_id: str = Field(min_length=1)
    situation_id: str = Field(min_length=1)


class BlueprintRowDraft(StrictModel):
    facet_id: str = Field(min_length=1)
    behavior_id: str = Field(min_length=1)
    candidate_references: list[BlueprintCandidateReference] = Field(
        min_length=INCREMENTAL_CANDIDATES_PER_CELL,
        max_length=INCREMENTAL_CANDIDATES_PER_CELL,
    )


class BlueprintAgentOutput(StrictModel):
    rows: list[BlueprintRowDraft] = Field(min_length=1)


def _agent(prompt: str, output_type: type, prefix: str, temperature: float) -> Any:
    from sjt_system.agent.agent_factory import create_agent

    model_id = os.getenv(f"{prefix}_MODEL_ID") or None
    value = float(os.getenv(f"{prefix}_TEMPERATURE", str(temperature)))
    return create_agent(prompt, output_type, model_id=model_id, temperature=value)


def create_expansion_agent() -> Any:
    return _agent(EXPANSION_PROMPT, FacetExpansionAgentOutput, "EXPANSION", 0.5)


def create_blueprint_agent() -> Any:
    return _agent(BLUEPRINT_PROMPT, BlueprintAgentOutput, "BLUEPRINT", 0.3)


def assign_expansion_ids(
    facet_id: str,
    target_population: str,
    output: FacetExpansionAgentOutput | Mapping[str, Any],
) -> FacetExpansion:
    result = (
        output
        if isinstance(output, FacetExpansionAgentOutput)
        else FacetExpansionAgentOutput.model_validate(output)
    )
    expansions = []
    for behavior in result.behavior_expansions:
        mechanisms = []
        for mechanism_index, mechanism in enumerate(behavior.mechanisms, start=1):
            mechanism_id = f"{behavior.behavior_id}_M{mechanism_index:02d}"
            mechanisms.append(
                MechanismRecord(
                    mechanism_id=mechanism_id,
                    activation_mechanism=mechanism.activation_mechanism,
                    situations=[
                        SituationRecord(
                            situation_id=f"{mechanism_id}_S{index:02d}",
                            **situation.model_dump(),
                        )
                        for index, situation in enumerate(
                            mechanism.situations, start=1
                        )
                    ],
                )
            )
        expansions.append(
            BehaviorExpansionRecord(
                behavior_id=behavior.behavior_id,
                mechanisms=mechanisms,
            )
        )
    return FacetExpansion(
        facet_id=facet_id,
        target_population=target_population,
        behavior_expansions=expansions,
    )


def expansion_cache_path(run_id: str, facet_id: str) -> Path:
    from sjt_system.runtime.output_paths import scoped_output
    return scoped_output("run_knowledge", DEFAULT_RUN_KNOWLEDGE_ROOT) / run_id / "expansions" / f"{facet_id}.json"


def load_facet_expansion(path: Path | str) -> FacetExpansion:
    return FacetExpansion.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


async def ensure_facet_expansion(
    *,
    run_id: str,
    facet: Mapping[str, Any],
    behavior_evidence: list[Mapping[str, Any]],
    target_population: str,
    output_language: str,
    required_situation_count: int = 1,
    agent: Any | None = None,
) -> FacetExpansion:
    from sjt_system.authoring.behavior_repertoire import (
        domain_coverage_problems, domain_coverage_rule, life_domains_for, load_manifestations,
        load_trigger_types, trigger_coverage_problems, trigger_coverage_rule,
    )
    required_situation_count = max(1, int(required_situation_count))
    path = expansion_cache_path(run_id, str(facet["facet_id"]))
    if path.exists():
        cached = load_facet_expansion(path)
        # 数量满足“至少”即可：蓝图从情境池中选取所需条目，池子偏大不影响
        if expansion_situation_count(cached) >= required_situation_count:
            return cached
    runnable = agent or create_expansion_agent()
    manifestations = load_manifestations(str(facet["facet_id"]), behavior_evidence)
    domains = life_domains_for(target_population)
    coverage = domain_coverage_rule(required_situation_count, len(domains))
    triggers = load_trigger_types(str(facet["facet_id"]), behavior_evidence)
    trigger_rule = trigger_coverage_rule(required_situation_count, len(triggers))
    last_count = 0
    coverage_problems: list[str] = []
    for attempt in range(_MAX_VALIDATION_RETRIES + 1):
        revision = ""
        if attempt and last_count < required_situation_count:
            revision = (
                f"上一候选有 {last_count} 个情境条目，但至少需要 "
                f"{required_situation_count} 个。保持现有 Behavior Evidence 和"
                "实质机制边界；如果数量不足，增加真实不同的情境条目；"
                "不要为凑数创建重复机制或同质情境。"
            )
        elif attempt and coverage_problems:
            revision = ("上一候选的生活领域或触发方式覆盖不符合要求：" + "；".join(coverage_problems)
                        + "。保持机制不变，调整情境的 domain、trigger_id 与事件，使其分布到更多生活领域和触发方式。")
        expansion = await _generate_expansion(
            runnable,
            facet,
            behavior_evidence,
            target_population,
            output_language,
            required_situation_count,
            extra_context=revision,
            behavior_manifestations=manifestations,
            life_domains=domains,
            domain_coverage=coverage,
            trigger_types=triggers,
            trigger_coverage=trigger_rule,
        )
        last_count = expansion_situation_count(expansion)
        if last_count < required_situation_count:
            continue
        coverage_problems = domain_coverage_problems(
            [situation.domain for behavior in expansion.behavior_expansions
             for mechanism in behavior.mechanisms for situation in mechanism.situations],
            domains, coverage)
        if triggers:
            entries = [(situation.trigger_id, behavior.behavior_id)
                       for behavior in expansion.behavior_expansions
                       for mechanism in behavior.mechanisms for situation in mechanism.situations]
            coverage_problems += trigger_coverage_problems(
                [tid for tid, _ in entries], triggers, trigger_rule, [bid for _, bid in entries])
        if coverage_problems and attempt < _MAX_VALIDATION_RETRIES:
            continue
        # 数量满足即接受；领域/触发方式覆盖在重试后仍不足时保留最后一版（软约束）。
        break
    else:
        raise ValueError(
            f"Behavior Expansion 最终提供 {last_count} 个情境条目，"
            f"但至少需要 {required_situation_count} 个"
        )
    expected_behavior_ids = {
        str(row["behavior_id"]) for row in behavior_evidence
    }
    actual_behavior_ids = [
        row.behavior_id for row in expansion.behavior_expansions
    ]
    if (
        set(actual_behavior_ids) != expected_behavior_ids
        or len(actual_behavior_ids) != len(expected_behavior_ids)
    ):
        raise ValueError("Expansion 必须逐一引用当前 facet 的行为证据")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, expansion.model_dump(mode="json"))
    return expansion


def expansion_situation_count(expansion: FacetExpansion) -> int:
    return sum(
        len(mechanism.situations)
        for behavior in expansion.behavior_expansions
        for mechanism in behavior.mechanisms
    )


async def _generate_expansion(
    runnable: Any,
    facet: Mapping[str, Any],
    behavior_evidence: list[Mapping[str, Any]],
    target_population: str,
    output_language: str,
    required_situation_count: int,
    extra_context: str = "",
    behavior_manifestations: list[Mapping[str, Any]] | None = None,
    life_domains: list[str] | None = None,
    domain_coverage: Mapping[str, int] | None = None,
    trigger_types: list[Mapping[str, Any]] | None = None,
    trigger_coverage: Mapping[str, int] | None = None,
) -> FacetExpansion:
    from sjt_system.authoring.behavior_repertoire import manifestations_for_prompt, trigger_types_for_prompt
    input_data: dict[str, Any] = {
        "facet_profile": dict(facet),
        "behavior_evidence": [dict(row) for row in behavior_evidence],
        "target_population": target_population,
        "output_language": output_language,
        "required_situation_count": required_situation_count,
    }
    if behavior_manifestations:
        input_data["behavior_manifestations"] = manifestations_for_prompt(behavior_manifestations)
    if life_domains:
        input_data["life_domains"] = list(life_domains)
        input_data["domain_coverage"] = dict(domain_coverage or {})
    if trigger_types:
        input_data["trigger_types"] = trigger_types_for_prompt(trigger_types)
        input_data["trigger_coverage"] = dict(trigger_coverage or {})
    if extra_context:
        input_data["revision_instruction"] = extra_context
    raw = await ainvoke_model_with_schema_repair(
        runnable,
        {"input_data": input_data},
        job_label=f"行为情境扩展-{facet['facet_id']}",
    )
    return assign_expansion_ids(
        str(facet["facet_id"]), target_population, raw
    )


def _neighbor_facet_definitions(
    target_facet: Mapping[str, Any],
) -> list[dict[str, Any]]:
    confounds = target_facet.get("common_confounds") or []
    if isinstance(confounds, str):
        confounds = [confounds]
    ambiguous_contexts = target_facet.get("inappropriate_conditions") or []
    if isinstance(ambiguous_contexts, str):
        ambiguous_contexts = [ambiguous_contexts]
    return [
        {
            "note": f"以下内容摘自目标 facet 的 common_confounds 和 inappropriate_conditions，"
                    f"是已知的可能与 {target_facet.get('facet_id', '')} 混淆的构念领域",
            "items": [str(item) for item in [*confounds, *ambiguous_contexts]],
        }
    ]


def _build_validation_feedback(
    failed: list[tuple[int, int, str, str]],
) -> str:
    lines = [
        "以下 activation mechanism 未能通过构念纯度校验，"
        "它们描述的行为差异更可能由相邻 facet 而非目标 facet 驱动。"
        "请重写这些 mechanism，确保核心行为差异由目标 facet 驱动。",
        "",
    ]
    for bi, mi, text, reason in failed:
        lines.append(f"失败的 mechanism: {text}")
        lines.append(f"校验判断: {reason}")
        lines.append("")
    lines.append(
        "修改要点：改变行为差异的载体，使高-低差异体现在的目标 facet 的核心行为维度上，"
        "而不是相邻 facet 的行为维度。"
    )
    return "\n".join(lines)


def _discard_failed_mechanisms(
    expansion: FacetExpansion,
    failed: list[tuple[int, int, str, str]],
) -> None:
    discarded_indices: dict[int, set[int]] = {}
    for bi, mi, _text, _reason in failed:
        discarded_indices.setdefault(bi, set()).add(mi)

    for bi in sorted(discarded_indices, reverse=True):
        behavior = expansion.behavior_expansions[bi]
        surviving = [
            deepcopy(m) for mi, m in enumerate(behavior.mechanisms)
            if mi not in discarded_indices.get(bi, set())
        ]
        if not surviving:
            raise ValueError(
                f"behavior_id={behavior.behavior_id} 的所有 mechanism "
                f"均未通过构念纯度校验且已达重试上限。"
                f"请检查 behavior evidence 与 facet 定义是否一致，"
                f"或人工审核该 facet 的行为证据。"
            )
        expansion.behavior_expansions[bi] = BehaviorExpansionRecord(
            behavior_id=behavior.behavior_id,
            mechanisms=surviving,
        )


async def propose_blueprint_rows(
    *,
    profile: Mapping[str, Any],
    expansions: list[FacetExpansion],
    generation_total: int,
    retention_total: int,
    facet_retention_quotas: Mapping[str, int],
    retry_feedback: str = "",
    agent: Any | None = None,
) -> BlueprintAgentOutput:
    runnable = agent or create_blueprint_agent()
    available_reference_count = sum(
        len(mechanism.situations)
        for expansion in expansions
        for behavior in expansion.behavior_expansions
        for mechanism in behavior.mechanisms
    )
    if available_reference_count < generation_total:
        raise ValueError(
            "Behavior Expansion 的唯一情境引用不足："
            f"需要 {generation_total} 个，实际 {available_reference_count} 个"
        )
    row_total = retention_total
    if row_total < 1:
        raise ValueError("Behavior Expansion 没有可用于蓝图的情境引用")
    input_data: dict[str, Any] = {
        "facets": [
            {
                "facet_id": facet["facet_id"],
                "facet_name": facet["facet_name"],
                "behavior_evidence": [
                    {
                        "behavior_id": ev["behavior_id"],
                        "behavior_dimension": ev["behavior_dimension"],
                        "high_expression": ev["high_expression"],
                        "low_expression": ev["low_expression"],
                        "boundary_condition": ev["boundary_condition"],
                    }
                    for ev in (facet.get("behavior_evidence") or [])
                ],
            }
            for facet in profile.get("facets") or []
        ],
        "expansions": [row.model_dump(mode="json") for row in expansions],
        "generation_total": generation_total,
        "retention_total": retention_total,
        "row_total": row_total,
        "candidate_count_per_row": INCREMENTAL_CANDIDATES_PER_CELL,
        "facet_retention_quotas": {
            str(facet_id): int(quota)
            for facet_id, quota in facet_retention_quotas.items()
        },
    }
    if retry_feedback:
        input_data["blueprint_semantic_feedback"] = retry_feedback
    raw = await ainvoke_model_with_schema_repair(
        runnable,
        {
            "input_data": input_data,
        },
        job_label="双向细目表设计",
    )
    return BlueprintAgentOutput.model_validate(raw)
