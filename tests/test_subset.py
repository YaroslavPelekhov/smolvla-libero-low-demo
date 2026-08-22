from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from vla_budget.subset import aggregate_episode_stats, ensure_subset_overlay


def test_aggregate_episode_stats_uses_only_requested_rows(tmp_path: Path) -> None:
    episodes_dir = tmp_path / "meta" / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True)
    table = pa.table(
        {
            "episode_index": [3, 4, 5],
            "stats/action/min": [[0.0], [2.0], [100.0]],
            "stats/action/max": [[2.0], [6.0], [200.0]],
            "stats/action/mean": [[1.0], [4.0], [150.0]],
            "stats/action/std": [[1.0], [2.0], [25.0]],
            "stats/action/count": [[2], [6], [100]],
            "stats/action/q50": [[1.0], [4.0], [150.0]],
        }
    )
    pq.write_table(table, episodes_dir / "file-000.parquet")

    stats = aggregate_episode_stats(tmp_path, [3, 4])

    np.testing.assert_allclose(stats["action"]["mean"], [3.25])
    np.testing.assert_allclose(stats["action"]["count"], [8])
    np.testing.assert_allclose(stats["action"]["min"], [0.0])
    np.testing.assert_allclose(stats["action"]["max"], [6.0])
    assert "q50" not in stats["action"]
    assert stats["action"]["mean"][0] < 10  # episode 5 must not leak into the subset


def test_overlay_can_anchor_stats_without_changing_training_subset(tmp_path: Path) -> None:
    source = tmp_path / "target"
    episodes_dir = source / "meta" / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True)
    (source / "data").mkdir()
    table = pa.table(
        {
            "episode_index": [7],
            "stats/action/min": [[0.0]],
            "stats/action/max": [[2.0]],
            "stats/action/mean": [[1.0]],
            "stats/action/std": [[0.5]],
            "stats/action/count": [[4]],
        }
    )
    pq.write_table(table, episodes_dir / "file-000.parquet")
    (source / "meta" / "info.json").write_text("{}", encoding="utf-8")

    anchor = tmp_path / "seen"
    (anchor / "meta").mkdir(parents=True)
    anchor_stats = {"action": {"count": [100], "mean": [3.0], "std": [2.0]}}
    (anchor / "meta" / "stats.json").write_text(
        json.dumps(anchor_stats), encoding="utf-8"
    )

    output = ensure_subset_overlay(
        source, tmp_path / "overlay", [7], stats_source_root=anchor
    )
    manifest = json.loads((output / ".vla_subset.json").read_text(encoding="utf-8"))
    written_stats = json.loads((output / "meta" / "stats.json").read_text(encoding="utf-8"))

    assert written_stats == anchor_stats
    assert manifest["training_frame_count"] == 4
    assert manifest["normalization_frame_count"] == 100
