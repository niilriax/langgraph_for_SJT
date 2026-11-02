"""Unattended small smoke run of the main workflow, stopping after N measurements.

Usage:
    python -X utf8 tools/run_smoke_test.py --run-id <uuid>            # fresh run
    python -X utf8 tools/run_smoke_test.py --run-id <uuid> --resume   # continue

Operator decisions are supplied automatically; any unexpected decision point
pauses the run and saves the payload instead of guessing.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import traceback
from uuid import UUID

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

SMOKE = {
    # At least two facets: item/form Δmin needs a non-target selected facet.
    "facet_item_counts": {
        "extraversion_gregariousness": 16,
        "neuroticism_self_consciousness": 16,
    },
    "target_population": "大学生",
    "sample_size_per_condition": 100,
    "seed": 7,
    "respondent_model_id": "deepseek-v4.1-flash",
    # Thinking mode on DeepSeek endpoints ignores sampling temperature; keep it
    # off for respondents so the fixed temperature=1.5 actually applies.
    "respondent_thinking": "disabled",
    "stop_after_measurements": 6,
    "repair_knowledge_mode": "run_only",
}

# Must be set before sjt_system loads .env (load_dotenv does not override).
os.environ["VIRTUAL_RESPONDENT_MODEL_ID"] = SMOKE["respondent_model_id"]
os.environ["VIRTUAL_RESPONDENT_THINKING"] = SMOKE["respondent_thinking"]

from sjt_system.authoring.construct_registry import (  # noqa: E402
    resolve_construct_selection,
)
from sjt_system.authoring.requirements import (  # noqa: E402
    SPECIFICATION_FIELDS,
    validate_test_specification,
)
from sjt_system.runtime.checkpoint import (  # noqa: E402
    DEFAULT_CHECKPOINT_ROOT,
    load_run_checkpoint,
    prepare_resumed_state,
    resolve_run_checkpoint_path,
    save_run_checkpoint,
)
from sjt_system.runtime.io import write_json_atomic  # noqa: E402
from sjt_system.state import create_initial_state  # noqa: E402

TASK_ROOT = PROJECT_ROOT / "outputs" / "smoke_tests"


class SmokeStop(Exception):
    """Planned stop after the configured number of measurements."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fresh_state(run_id: str) -> dict:
    counts = dict(SMOKE["facet_item_counts"])
    selection = {"inventory_id": "neo_pi_r",
                 "domain_ids": list(dict.fromkeys(facet.split("_", 1)[0] for facet in counts)),
                 "facet_ids": list(counts)}
    profile = resolve_construct_selection(selection)
    total = sum(counts.values())
    spec = {
        "construct_selection": selection,
        "target_population": SMOKE["target_population"],
        "facet_item_counts": counts,
        "final_item_count": total,
        "output_language": "zh-CN",
    }
    errors = validate_test_specification(spec)
    if errors:
        raise ValueError(f"Smoke specification invalid: {errors}")
    state = create_initial_state(
        f"冒烟测试：为{SMOKE['target_population']}开发{total}题PSJT（{'、'.join(counts)}）",
        repair_knowledge_mode=SMOKE["repair_knowledge_mode"],
    )
    state.update(
        run_id=run_id,
        test_specification=spec,
        construct_profile=profile,
        failure_call_replenishment=True,
        specification_sources={key: "user" for key in SPECIFICATION_FIELDS},
        requirements_confirmed=True,
        confirmed_requirement_fields=sorted(SPECIFICATION_FIELDS),
    )
    return state


def operator_decision(payload: dict, task_dir: Path) -> dict:
    kind = payload.get("type")
    if payload.get("validation_error"):
        write_json_atomic(task_dir / "operator_interrupt.json", payload)
        raise RuntimeError(f"Operator decision failed validation: {payload['validation_error']}")
    if kind == "repair_knowledge_selection":
        decision = {"mode": SMOKE["repair_knowledge_mode"]}
    elif kind == "item_development_mode_selection":
        decision = {"mode": "automatic"}
    elif kind == "virtual_sample_selection":
        decision = {
            "sample_size_per_condition": SMOKE["sample_size_per_condition"],
            "seed": SMOKE["seed"],
            "max_concurrency": 0,
            "max_retries": 4,  # system default: up to 5 attempts
            "score_distribution": {"family": "normal", "mean": 50.0, "sd": 15.0},
        }
    elif kind == "post_virtual_response_decision":
        history = payload.get("psychometric_iteration_history") or []
        controller = (history[-1].get("facet_iteration_state") or {}) if history else {}
        # A completed controller proceeds to assembly and reports even at the cap.
        if controller.get("status") != "complete" and len(history) >= SMOKE["stop_after_measurements"]:
            decision = {"decision": "stop"}
        else:
            decision = {"decision": "start"}
    elif kind == "agent_result_approval" and "approve" in payload.get("available_decisions", []):
        decision = {"decision": "approve", "feedback": None}
    else:
        write_json_atomic(task_dir / "operator_interrupt.json", payload)
        raise RuntimeError(f"Unconfigured operator decision: {kind!r}, action={payload.get('action')!r}")
    print(f"[operator] {kind}: {json.dumps(decision, ensure_ascii=False)}", flush=True)
    with (task_dir / "operator_decisions.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"recorded_at": now(), "type": kind, "action": payload.get("action"),
                                 "decision": decision}, ensure_ascii=False) + "\n")
    return decision


async def execute(state: dict, task_dir: Path) -> int:
    import cli_app
    from sjt_system.workflow.graph import build_sjt_graph

    cli_app.app = build_sjt_graph()
    cli_app.prompt_user_decision = lambda payload: operator_decision(payload, task_dir)
    try:
        result = await cli_app.run_with_trace(state, checkpoint_root=DEFAULT_CHECKPOINT_ROOT)
    except Exception as exc:
        traceback.print_exc()
        checkpoint = resolve_run_checkpoint_path(state["run_id"], DEFAULT_CHECKPOINT_ROOT)
        result = load_run_checkpoint(checkpoint)["state"] if checkpoint.is_file() else state
        write_json_atomic(task_dir / "outcome.json", {"status": "error", "error": str(exc),
                                                      "recorded_at": now()})
        return 2
    save_run_checkpoint(result, checkpoint_root=DEFAULT_CHECKPOINT_ROOT)
    write_json_atomic(task_dir / "outcome.json", {
        "status": result.get("status"), "current_phase": result.get("current_phase"),
        "measurements": len(result.get("psychometric_iteration_history") or []),
        "item_count": len(result.get("item_pool") or []),
        "latest_errors": (result.get("errors") or [])[-3:], "recorded_at": now(),
    })
    print(f"[smoke] finished with status={result.get('status')}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-measurements", type=int,
                        help="override the measurement cap (e.g. to finish a capped run on resume)")
    parser.add_argument("--facets", help="comma-separated facet ids (fresh runs only); "
                        "each gets the same item count as the default config")
    parser.add_argument("--sample-size", type=int,
                        help="virtual respondents per group (fresh runs only; default: the config value)")
    parser.add_argument("--items-per-facet", type=int,
                        help="item count per facet (fresh runs only; default: the config value)")
    args = parser.parse_args()
    if args.facets:
        per_facet = args.items_per_facet or next(iter(SMOKE["facet_item_counts"].values()))
        SMOKE["facet_item_counts"] = {facet.strip(): per_facet for facet in args.facets.split(",") if facet.strip()}
    if args.sample_size:
        SMOKE["sample_size_per_condition"] = args.sample_size
    if args.max_measurements:
        SMOKE["stop_after_measurements"] = args.max_measurements
    UUID(args.run_id)
    task_dir = TASK_ROOT / args.run_id
    task_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(PROJECT_ROOT)
    checkpoint = resolve_run_checkpoint_path(args.run_id, DEFAULT_CHECKPOINT_ROOT)
    if args.resume:
        state = prepare_resumed_state(load_run_checkpoint(checkpoint)["state"])
    else:
        if checkpoint.exists():
            raise ValueError("Run already exists; pass --resume to continue it")
        state = fresh_state(args.run_id)
        write_json_atomic(task_dir / "smoke_config.json", {**SMOKE, "run_id": args.run_id,
                                                           "created_at": now(),
                                                           "authoring_model_id": os.getenv("MODEL_ID")})
    return asyncio.run(execute(state, task_dir))


if __name__ == "__main__":
    raise SystemExit(main())
