from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime

from .config import load_config, resolve
from .metadata import load_suite_metadata


def manifest_digest(payload: dict) -> str:
    identity = {
        key: value for key, value in payload.items() if key not in {"created_at_utc", "sha256"}
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def build_manifest(cfg: dict) -> dict:
    data = cfg["dataset"]
    base = resolve(data["local_dir"])
    seen = load_suite_metadata(base / data["seen_suite"])
    target = load_suite_metadata(base / data["target_suite"])
    target_ids = [int(value) for value in data["target_task_ids"]]
    configured_prompts = {
        int(task_id): str(prompt) for task_id, prompt in data["target_prompts"].items()
    }
    budgets = [int(value) for value in cfg["protocol"]["budgets"]]
    maximum = max(budgets)

    if seen.total_tasks < 50:
        raise ValueError(
            f"Seen suite has only {seen.total_tasks} tasks; this is not a LIBERO-90-compatible source"
        )
    missing = set(target_ids) - set(configured_prompts)
    if missing:
        raise ValueError(f"Target task ids absent from canonical prompt mapping: {sorted(missing)}")
    prompt_to_dataset_id = {prompt: task_id for task_id, prompt in target.task_prompts.items()}
    absent_prompts = {
        configured_prompts[task_id]
        for task_id in target_ids
        if configured_prompts[task_id] not in prompt_to_dataset_id
    }
    if absent_prompts:
        raise ValueError(f"Canonical simulator prompts absent from dataset: {sorted(absent_prompts)}")
    overlap = set(seen.task_prompts.values()) & {
        configured_prompts[task_id] for task_id in target_ids
    }
    if overlap:
        raise ValueError(f"Seen/held-out language leakage detected: {sorted(overlap)}")

    tasks = {}
    for task_id in target_ids:
        prompt = configured_prompts[task_id]
        dataset_task_id = prompt_to_dataset_id[prompt]
        ordered = list(target.episodes_by_task[dataset_task_id])
        if len(ordered) < maximum:
            raise ValueError(f"Task {task_id} has {len(ordered)} episodes, fewer than {maximum}")
        tasks[str(task_id)] = {
            "prompt": prompt,
            "dataset_task_index": dataset_task_id,
            "wrong_prompt_task_id": target_ids[(target_ids.index(task_id) + 1) % len(target_ids)],
            "episodes": {str(budget): ordered[:budget] for budget in budgets},
        }

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_repo_id": data["repo_id"],
        "dataset_revision": data.get("revision", "main"),
        "seen": {
            "suite": data["seen_suite"],
            "total_tasks": seen.total_tasks,
            "total_episodes": seen.total_episodes,
        },
        "target": {"suite": data["target_suite"], "tasks": tasks},
        "selection_rule": (
            "map simulator task_id to dataset task_index by exact prompt, then take first "
            "episodes in ascending episode_index"
        ),
    }
    payload["sha256"] = manifest_digest(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and validate the fixed demo manifest")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--data-root",
        help="Override dataset.local_dir without changing the frozen experiment config",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.data_root:
        cfg["dataset"]["local_dir"] = args.data_root
    manifest = build_manifest(cfg)
    output = resolve(cfg["paths"]["manifest"])
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        old_identity = {
            key: value for key, value in existing.items() if key not in {"created_at_utc", "sha256"}
        }
        new_identity = {
            key: value for key, value in manifest.items() if key not in {"created_at_utc", "sha256"}
        }
        if old_identity != new_identity:
            raise SystemExit("Refusing to overwrite a changed protocol manifest")
        manifest["created_at_utc"] = existing["created_at_utc"]
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {output} ({manifest['sha256']})")


if __name__ == "__main__":
    main()
