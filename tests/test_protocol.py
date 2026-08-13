import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from vla_budget.protocol import build_manifest, manifest_digest


def _write_suite(root: Path, prompts: list[str], episodes_per_task: int) -> None:
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "total_episodes": len(prompts) * episodes_per_task,
                "total_tasks": len(prompts),
            }
        ),
        encoding="utf-8",
    )
    tasks = pa.table(
        {
            "task_index": list(range(len(prompts))),
            "__index_level_0__": prompts,
        }
    )
    pq.write_table(tasks, root / "meta" / "tasks.parquet")
    rows = []
    episode = 0
    # Interleave task ids to prove selection follows episode_index, not contiguous ranges.
    for _ in range(episodes_per_task):
        for prompt in prompts:
            rows.append({"episode_index": episode, "tasks": [prompt]})
            episode += 1
    pq.write_table(
        pa.Table.from_pylist(rows), root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )


def test_manifest_uses_nested_first_prefixes(tmp_path):
    base = tmp_path / "data"
    _write_suite(base / "libero_90", [f"seen {i}" for i in range(60)], 1)
    _write_suite(base / "libero_goal", [f"target {i}" for i in range(3)], 30)
    cfg = {
        "dataset": {
            "repo_id": "test/libero",
            "revision": "abc",
            "local_dir": str(base),
            "seen_suite": "libero_90",
            "target_suite": "libero_goal",
            "target_task_ids": [0, 1, 2],
            # Reverse the dataset task order to prove selection is matched by prompt.
            "target_prompts": {0: "target 2", 1: "target 1", 2: "target 0"},
        },
        "protocol": {"budgets": [5, 10, 25]},
    }
    manifest = build_manifest(cfg)
    task = manifest["target"]["tasks"]["1"]["episodes"]
    assert task["5"] == [1, 4, 7, 10, 13]
    assert task["10"][:5] == task["5"]
    assert task["25"][:10] == task["10"]
    assert len(manifest["sha256"]) == 64
    assert manifest["target"]["tasks"]["0"]["dataset_task_index"] == 2
    changed_time = dict(manifest, created_at_utc="later")
    assert manifest_digest(changed_time) == manifest["sha256"]
