"""Concurrent fixed-slot development using the existing single-item nodes."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import quote

from sjt_system.runtime.io import write_json_atomic
from sjt_system.runtime.output_paths import scoped_output
from sjt_system.runtime.progress import emit_progress


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MERGED_MAPS = (
    "item_skeletons", "skeleton_reviews", "skeleton_review_history",
    "skeleton_failures", "item_history", "item_pattern_profiles", "item_lineage",
    "item_content_evidence",
)


SKELETON_PLAN_ATTEMPTS = 3
DIVERSITY_REWRITE_ATTEMPTS = 2
MAX_DIVERSITY_REWRITES_PER_FACET = 8


def _anchor_summary(skeleton: Mapping[str, Any], channel: str | None) -> dict[str, Any]:
    tendencies = {str(row.get("behavioral_level")): row.get("behavioral_tendency")
                  for row in skeleton.get("option_structure") or [] if isinstance(row, Mapping)}
    return {"difference_channel": channel, "situation_type": skeleton.get("situation_type"),
            "low": tendencies.get("low"), "high": tendencies.get("high")}


async def plan_facet_skeletons(
    state: Mapping[str, Any], slots: list[tuple[dict, dict]], plan_path: Path,
) -> dict[str, dict[str, Any]]:
    """Plan the skeletons of each facet's pending slots before item writing.

    Facets run concurrently; within a facet skeletons are generated in order so
    each one sees its siblings' anchors and gets its own difference channel."""
    from sjt_system.authoring.behavior_repertoire import ManifestationAssigner, load_manifestations
    from sjt_system.authoring.diversity import (
        difference_channels_for, label_words_for, skeleton_label_problems,
    )
    from sjt_system.authoring.generation_plan import _expansion_models
    from sjt_system.workflow.executor import _fill_compact_slots

    blueprint = state["blueprint"]
    existing = {str(key): value for key, value in (state.get("item_skeletons") or {}).items()}
    plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else {}
    skeletons = plan.setdefault("skeletons", {})
    channels = plan.setdefault("channels", {})
    behaviors = plan.setdefault("behaviors", {})
    profile_facets = {str(row["facet_id"]): row for row in (state.get("construct_profile") or {}).get("facets") or []}
    try:
        expansions = _expansion_models(blueprint)
    except Exception:  # noqa: BLE001 - behaviors are optional guidance
        expansions = []
    situation_behaviors = {situation.situation_id: list(situation.manifestation_ids)
                           for expansion in expansions for behavior in expansion.behavior_expansions
                           for mechanism in behavior.mechanisms for situation in mechanism.situations}
    lock = asyncio.Lock()
    by_facet: dict[str, list[tuple[dict, dict]]] = {}
    for cell, specification in slots:
        by_facet.setdefault(str(cell["facet_id"]), []).append((cell, specification))

    async def plan_facet(facet_id: str, rows: list[tuple[dict, dict]]) -> None:
        rotation = difference_channels_for(facet_id, len(rows))
        words = label_words_for(facet_id)
        assigner = ManifestationAssigner(load_manifestations(
            facet_id, (profile_facets.get(facet_id) or {}).get("behavior_evidence") or []))
        for assigned in behaviors.values():
            assigner.note(assigned.get("high"))
            assigner.note(assigned.get("low"))
        anchors: list[dict[str, Any]] = []
        for index, (cell, specification) in enumerate(rows):
            item_id = str(specification["specification_id"])
            known = existing.get(item_id) or skeletons.get(item_id)
            if known:
                anchors.append(_anchor_summary(known, channels.get(item_id)))
                continue
            channel = rotation[index]
            preferred = situation_behaviors.get(str(cell.get("situation_id")), [])
            high = assigner.pick(str(cell.get("behavior_id")), "high", preferred)
            low = assigner.pick(str(cell.get("behavior_id")), "low", preferred)
            assigned_behaviors = ({"high": high["description"], "low": low["description"]}
                                  if high and low else None)
            feedback = None
            skeleton = None
            for attempt in range(1, SKELETON_PLAN_ATTEMPTS + 1):
                extra = {"assigned_difference_channel": channel,
                         "sibling_skeleton_anchors": deepcopy(anchors),
                         "facet_label_words": words}
                if assigned_behaviors:
                    extra["assigned_behaviors"] = assigned_behaviors
                if feedback:
                    extra["planning_feedback"] = feedback
                try:
                    generated, _ = await _fill_compact_slots(
                        state, blueprint, cell, target_ids={item_id}, extra_input=extra)
                except ValueError:
                    break
                skeleton = generated[item_id]
                problems = skeleton_label_problems(skeleton, words)
                if not problems:
                    break
                feedback = {"problems": problems, "attempt": attempt + 1}
            if skeleton is None:
                # Leave the slot to its own microflow, which reports failures normally.
                continue
            async with lock:
                skeletons[item_id] = skeleton
                channels[item_id] = channel
                if high and low:
                    behaviors[item_id] = {"high": high["manifestation_id"], "low": low["manifestation_id"]}
                write_json_atomic(plan_path, plan)
            anchors.append(_anchor_summary(skeleton, channel))

    await asyncio.gather(*(plan_facet(facet_id, rows) for facet_id, rows in by_facet.items()))
    return {key: deepcopy(value) for key, value in skeletons.items()}


async def diversify_options(
    state: Mapping[str, Any], update: dict[str, Any], log_path: Path,
) -> None:
    """Rewrite items whose options share a template with too many siblings.

    Facets run concurrently; inside a facet flagged items are rewritten one by
    one so every rewrite sees the current siblings. Only the scenario and
    option texts change; option identity, behavioral levels and scoring stay."""
    from sjt_system.authoring.context import build_item_pattern_profile
    from sjt_system.authoring.diversity import option_diversity_context, option_template_report
    from sjt_system.workflow.executor import execute_item_action_with_repair

    merged = {**state, **update}
    pool = {str(row["item_id"]): row for row in update["item_pool"]}
    specifications = {str(row["specification_id"]): row for row in update.get("item_specifications") or []}
    cells = {str(row["cell_id"]): row for row in (state.get("blueprint") or {}).get("cells") or []}
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {"rewrites": []}
    done = {row["item_id"] for row in log["rewrites"]}
    lock = asyncio.Lock()
    facets: dict[str, list[str]] = {}
    for item_id, item in pool.items():
        facets.setdefault(str(item.get("target_dimension_id")), []).append(item_id)

    async def fix_facet(item_ids: list[str]) -> None:
        def siblings():
            return [pool[item_id] for item_id in item_ids]
        targets = [item_id for item_id in option_template_report(siblings())["flagged"]
                   if item_id not in done][:MAX_DIVERSITY_REWRITES_PER_FACET]
        for item_id in targets:
            reasons = option_template_report(siblings())["flagged"].get(item_id)
            if not reasons or item_id not in specifications:
                continue
            original = pool[item_id]
            best, best_reasons = original, list(reasons)
            for _attempt in range(DIVERSITY_REWRITE_ATTEMPTS):
                local = {**merged, "item_pool": siblings(), "current_item": None,
                         "current_item_review": None, "active_psychometric_repair": None,
                         "current_item_specification": deepcopy(specifications[item_id]),
                         "current_blueprint_cell": deepcopy(cells[str(original["blueprint_cell_id"])]),
                         "option_diversity_context": option_diversity_context(siblings(), item_id, best_reasons)}
                route = {"next_action": "generate_item", "reason": "option diversity rewrite",
                         "target_item_id": item_id, "target_blueprint_cell_id": original["blueprint_cell_id"]}
                try:
                    result = await execute_item_action_with_repair("generate_item", route, local)
                except ValueError:
                    continue
                generated = result["state_update"]["current_item"]
                texts = {row["behavioral_level"]: row["text"] for row in generated["response_options"]}
                if set(texts) != {row["behavioral_level"] for row in original["response_options"]}:
                    continue
                candidate = {**deepcopy(original), "scenario": generated["scenario"],
                             "response_options": [{**deepcopy(row), "text": texts[row["behavioral_level"]]}
                                                  for row in original["response_options"]]}
                trial = [candidate if row["item_id"] == item_id else row for row in siblings()]
                trial_reasons = option_template_report(trial)["flagged"].get(item_id, [])
                if len(trial_reasons) < len(best_reasons):
                    best, best_reasons = candidate, trial_reasons
                    pool[item_id] = candidate
                if not trial_reasons:
                    break
            async with lock:
                if best is not original:
                    update["item_pattern_profiles"][item_id] = build_item_pattern_profile(
                        best, specifications[item_id])
                log["rewrites"].append({"item_id": item_id, "reasons_before": reasons,
                                        "reasons_after": best_reasons, "rewritten": best is not original})
                write_json_atomic(log_path, log)

    await asyncio.gather(*(fix_facet(item_ids) for item_ids in facets.values()))
    update["item_pool"] = list(pool.values())


def pending_slots(state: Mapping[str, Any]) -> list[tuple[dict, dict]]:
    blueprint = state.get("blueprint") or {}
    cells = {str(row["cell_id"]): row for row in blueprint.get("cells") or []}
    specifications = {
        str(row["specification_id"]): row for row in state.get("item_specifications") or []
    }
    occupied = {str(row["item_id"]) for row in state.get("item_pool") or []}
    consumed = set(state.get("item_generation_completed_ids") or [])
    explicit_slots = "item_generation_completed_ids" in state
    counts: dict[str, int] = {}
    result = []
    seen = set()
    for slot in blueprint.get("slots") or []:
        item_id = str(slot["specification_id"])
        if item_id in seen:
            raise ValueError(f"Duplicate fixed slot: {item_id}")
        seen.add(item_id)
        cell_id = str(slot["blueprint_cell_id"])
        ordinal = counts.get(cell_id, 0)
        counts[cell_id] = ordinal + 1
        generated = int((state.get("blueprint_progress") or {}).get(cell_id, {}).get("generated") or 0)
        if item_id in occupied or item_id in consumed or (not explicit_slots and ordinal < generated):
            continue
        cell = deepcopy(cells[cell_id])
        specification = deepcopy(specifications.get(item_id) or {
            "specification_id": item_id,
            "blueprint_cell_id": cell_id,
            "target_dimension_id": cell["facet_id"],
        })
        result.append((cell, specification))
    return result


async def _develop_slot(
    state: Mapping[str, Any], cell: dict, specification: dict,
    *, checkpoint_path: Path,
) -> dict[str, Any]:
    # Imports stay local because execute_node dispatches back through executor.
    from sjt_system.workflow.execution_node import execute_node
    from sjt_system.workflow.interaction_nodes import automatic_approval_node, commit_node
    from sjt_system.workflow.item_nodes import (
        abandon_item_node, accept_item_node, prepare_item_review_node,
        prepare_item_revision_node,
    )
    from sjt_system.workflow.routes import (
        route_after_commit, route_after_execute, route_after_prepare_item_review,
    )

    item_id = str(specification["specification_id"])
    source = {
        "protocol": "fixed_slot_concurrent_v1",
        "blueprint": state.get("blueprint"),
        "test_specification": state.get("test_specification"),
        "specification_id": item_id,
        "model_settings": {name: os.getenv(name) for name in (
            "MODEL_ID", "BASE_URL", "STRUCTURED_OUTPUT_METHOD",
            "SKELETON_GENERATION_MODEL_ID", "SKELETON_GENERATION_TEMPERATURE",
        )},
    }
    from sjt_system.evaluation.virtual_content_review import is_enabled
    if is_enabled(state):
        source["virtual_content_review_protocol"] = state["virtual_content_review_protocol"]
    local = {
        **deepcopy(dict(state)),
        "item_development_mode": "automatic",
        "current_blueprint_cell": cell,
        "current_item_specification": specification,
        "current_item": None,
        "current_item_review": None,
        "active_psychometric_repair": None,
        "current_item_repair_attempted": False,
        "current_item_repair_failure": None,
        "current_item_revision_count": 0,
        "current_item_rewrite_count": 0,
        "current_item_replacement_count": 0,
        "review_process_status": "not_started",
        "skeleton_slot_failure_pending": False,
        "current_skeleton_repair_required": False,
        "pending_action": None, "pending_state_update": None,
        "pending_interaction": None, "user_decision": None,
        "execution_history": [],
        "errors": [],
        "route": {
            "next_action": "generate_item", "reason": "Concurrent fixed-slot development",
            "target_item_id": item_id, "target_blueprint_cell_id": cell["cell_id"],
        },
    }
    node = "execute"
    baseline_progress = deepcopy((state.get("blueprint_progress") or {}).get(str(cell["cell_id"])) or {})
    if checkpoint_path.exists():
        saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if saved.get("source") != source:
            raise ValueError(f"Fixed-slot checkpoint inputs changed: {item_id}")
        local = saved["state"]
        node = saved["next_node"]
        baseline_progress = saved["baseline_progress"]
        if local.get("status") == "failed" and node == "execute":
            local["status"] = "running"
    local.pop("max_steps", None)
    initial_steps = int(local.get("step_count") or 0)
    consumed_steps = 0

    def save() -> None:
        write_json_atomic(checkpoint_path, {
            "source": source, "next_node": node, "state": local,
            "baseline_progress": baseline_progress,
        })

    emit_progress({"type": "item_batch_progress", "status": "started", "item_id": item_id})
    try:
        while node != "done":
            if node == "execute":
                consumed_steps += 1
                local.update(await execute_node(local))
                if local.get("status") == "failed":
                    save()
                    break
                if local.get("skeleton_slot_failure_pending"):
                    node = "done"
                else:
                    node = route_after_execute(local)
            elif node == "automatic_approval":
                local.update(automatic_approval_node(local))
                node = "commit"
            elif node == "commit":
                local.update(commit_node(local))
                node = route_after_commit(local)
            elif node == "review":
                local.update(prepare_item_review_node(local))
                node = route_after_prepare_item_review(local)
            elif node in {"revise", "retry_item"}:
                local.update(prepare_item_revision_node(local))
                node = "execute"
            elif node in {"accept", "accept_latest"}:
                local.update(accept_item_node(local))
                node = "done"
            elif node == "abandon":
                local.update(abandon_item_node(local))
                node = "done"
            else:
                raise RuntimeError(f"Unexpected fixed-slot transition: {node}")
            save()
    except Exception as exc:
        local["status"] = "failed"
        local["errors"].append({"action": "generate_items_batch", "item_id": item_id, "message": str(exc)})
        save()
    failed = local.get("status") == "failed"
    emit_progress({"type": "item_batch_progress", "status": "failed" if failed else "completed", "item_id": item_id})
    return {
        "item_id": item_id, "cell_id": str(cell["cell_id"]), "state": local,
        "consumed_steps": max(consumed_steps, int(local.get("step_count") or 0) - initial_steps),
        "failed": failed,
        "progress_delta": {
            key: int(count) - int(baseline_progress.get(key) or 0)
            for key, count in (local.get("blueprint_progress") or {}).get(str(cell["cell_id"]), {}).items()
        },
    }


async def execute_item_generation_batch(state: Mapping[str, Any]) -> dict[str, Any]:
    slots = pending_slots(state)
    pending_ids = {str(specification["specification_id"]) for _, specification in slots}
    completed_ids = set(state.get("item_generation_completed_ids") or [])
    if "item_generation_completed_ids" not in state:
        completed_ids.update(str(row["specification_id"]) for row in (state.get("blueprint") or {}).get("slots") or []
                             if str(row["specification_id"]) not in pending_ids)
    checkpoint_dir = scoped_output("item_generation", PROJECT_ROOT / "outputs" / "item_generation") / str(state["run_id"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    from sjt_system.evaluation.virtual_content_review import is_enabled as diversity_enabled
    slot_state = state
    if diversity_enabled(state):
        unstarted = [(cell, specification) for cell, specification in slots
                     if not (checkpoint_dir / (quote(str(specification["specification_id"]), safe="") + ".json")).exists()]
        planned = await plan_facet_skeletons(state, unstarted, checkpoint_dir / "skeleton_plan.json")
        slot_state = {**state, "item_skeletons": {**(state.get("item_skeletons") or {}), **planned}}
    outcomes = await asyncio.gather(*(
        _develop_slot(slot_state, cell, specification,
            checkpoint_path=checkpoint_dir / (quote(str(specification["specification_id"]), safe="") + ".json"))
        for cell, specification in slots
    ), return_exceptions=True)
    update: dict[str, Any] = {
        key: deepcopy(state.get(key) or {}) for key in MERGED_MAPS
    }
    pool = {str(row["item_id"]): deepcopy(row) for row in state.get("item_pool") or []}
    specifications = {str(row["specification_id"]): deepcopy(row) for row in state.get("item_specifications") or []}
    progress = deepcopy(state.get("blueprint_progress") or {})
    context_usage = deepcopy(state.get("context_usage") or {})
    errors = deepcopy(state.get("errors") or [])
    rejected = deepcopy(state.get("rejected_items") or [])
    history = deepcopy(state.get("execution_history") or [])
    failures = 0
    steps = int(state["step_count"])
    for (cell, specification), outcome in zip(slots, outcomes, strict=True):
        item_id = str(specification["specification_id"])
        if isinstance(outcome, BaseException):
            failures += 1
            errors.append({"action": "generate_items_batch", "item_id": item_id, "message": str(outcome)})
            continue
        local = outcome["state"]
        steps += outcome["consumed_steps"]
        failures += int(outcome["failed"])
        if not outcome["failed"]:
            completed_ids.add(item_id)
        for key in MERGED_MAPS:
            if item_id in (local.get(key) or {}):
                update[key][item_id] = deepcopy(local[key][item_id])
            elif key == "skeleton_reviews":
                update[key].pop(item_id, None)
        for row in local.get("item_specifications") or []:
            if str(row.get("specification_id")) == item_id:
                specifications[item_id] = deepcopy(row)
        for row in local.get("item_pool") or []:
            if str(row.get("item_id")) == item_id and item_id not in pool:
                pool[item_id] = deepcopy(row)
                category = (local.get("item_pattern_profiles") or {}).get(item_id, {}).get("context_category")
                if category:
                    context_usage[category] = context_usage.get(category, 0) + 1
        rejected.extend(deepcopy(row) for row in local.get("rejected_items") or []
                        if str(row.get("item_id")) == item_id)
        cell_id = str(cell["cell_id"])
        if not outcome["failed"]:
            counters = progress.setdefault(cell_id, {})
            for key, delta in outcome["progress_delta"].items():
                counters[key] = counters.get(key, 0) + delta
        errors.extend(deepcopy(local.get("errors") or []))
        for event in local.get("execution_history") or []:
            event = deepcopy(event)
            event["event_id"] = f"{event.get('event_id')}:{item_id}"
            event["batch_item_id"] = item_id
            history.append(event)
    update.update({
        "item_pool": list(pool.values()), "item_specifications": list(specifications.values()),
        "blueprint_progress": progress, "context_usage": context_usage,
        "item_generation_completed_ids": sorted(completed_ids),
        "errors": errors, "rejected_items": rejected, "execution_history": history,
        "step_count": steps + 1, "status": "failed" if failures else "running",
        "current_item": None, "current_item_specification": None,
        "current_blueprint_cell": None, "current_item_review": None,
        "current_item_repair_failure": None, "current_item_repair_attempted": False,
        "skeleton_slot_failure_pending": False, "review_process_status": "not_started",
    })
    if diversity_enabled(state) and not failures and slots:
        await diversify_options(state, update, checkpoint_dir / "option_diversity.json")
    return {"state_update": update, "summary": f"Concurrent fixed-slot batch: {len(slots)} slots, {failures} failures; progress saved per slot."}
