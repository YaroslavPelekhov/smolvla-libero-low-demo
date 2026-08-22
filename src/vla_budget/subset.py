from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

STAT_COLUMN = re.compile(r"^stats/(.+)/(min|max|mean|std|count|q\d+)$")


def _episode_rows(source_root: Path, episodes: list[int]) -> list[dict]:
    requested = set(episodes)
    rows: list[dict] = []
    for path in sorted((source_root / "meta" / "episodes").glob("*/*.parquet")):
        rows.extend(
            row
            for row in pq.read_table(path).to_pylist()
            if int(row["episode_index"]) in requested
        )
    found = {int(row["episode_index"]) for row in rows}
    if found != requested:
        raise ValueError(f"Episode metadata mismatch: requested={sorted(requested)}, found={sorted(found)}")
    if len(rows) != len(requested):
        raise ValueError("Duplicate episode metadata rows detected")
    return rows


def _row_stats(row: dict) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for column, value in row.items():
        match = STAT_COLUMN.match(column)
        if match:
            feature, statistic = match.groups()
            result.setdefault(feature, {})[statistic] = np.asarray(value, dtype=np.float64)
    return result


def _aggregate_feature(items: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    means = np.stack([item["mean"] for item in items])
    variances = np.stack([item["std"] ** 2 for item in items])
    counts = np.stack([item["count"] for item in items])
    total_count = counts.sum(axis=0)
    expanded_counts = counts
    while expanded_counts.ndim < means.ndim:
        expanded_counts = np.expand_dims(expanded_counts, axis=-1)
    total_mean = (means * expanded_counts).sum(axis=0) / total_count
    total_variance = ((variances + (means - total_mean) ** 2) * expanded_counts).sum(
        axis=0
    ) / total_count
    result = {
        "min": np.min(np.stack([item["min"] for item in items]), axis=0),
        "max": np.max(np.stack([item["max"] for item in items]), axis=0),
        "mean": total_mean,
        "std": np.sqrt(total_variance),
        "count": total_count,
    }
    # Per-episode quantiles cannot be merged exactly without the underlying samples.
    # SmolVLA uses MEAN_STD for state/action, so omit them instead of writing an
    # apparently precise but mathematically invalid approximation.
    return result


def aggregate_episode_stats(source_root: Path, episodes: list[int]) -> dict:
    per_episode = [_row_stats(row) for row in _episode_rows(source_root, episodes)]
    features = {feature for stats in per_episode for feature in stats}
    return {
        feature: _aggregate_feature([stats[feature] for stats in per_episode if feature in stats])
        for feature in sorted(features)
    }


def _serialize(value):
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def ensure_subset_overlay(
    source_root: Path,
    output_root: Path,
    episodes: list[int],
    *,
    stats_source_root: Path | None = None,
) -> Path:
    source_root = source_root.resolve()
    stats_source_root = stats_source_root.resolve() if stats_source_root else None
    episodes = [int(value) for value in episodes]
    if not episodes or len(episodes) != len(set(episodes)):
        raise ValueError("Subset episodes must be a non-empty unique list")
    identity = {
        "source_root": str(source_root),
        "episodes": episodes,
        "stats_source_root": str(stats_source_root) if stats_source_root else None,
    }
    manifest_path = output_root / ".vla_subset.json"
    if output_root.exists():
        if not manifest_path.exists():
            raise FileExistsError(f"Subset overlay exists without manifest: {output_root}")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if {key: existing[key] for key in identity} != identity:
            raise ValueError(f"Subset overlay identity mismatch: {output_root}")
        return output_root

    selected_stats = _serialize(aggregate_episode_stats(source_root, episodes))
    if stats_source_root:
        stats = json.loads(
            (stats_source_root / "meta" / "stats.json").read_text(encoding="utf-8")
        )
    else:
        stats = selected_stats
    output_root.mkdir(parents=True)
    for name in ("data", "videos"):
        source = source_root / name
        if source.exists():
            os.symlink(source, output_root / name, target_is_directory=True)
    meta_output = output_root / "meta"
    meta_output.mkdir()
    for source in (source_root / "meta").iterdir():
        if source.name != "stats.json":
            os.symlink(source, meta_output / source.name, target_is_directory=source.is_dir())
    stats_text = json.dumps(stats, indent=2, sort_keys=True) + "\n"
    (meta_output / "stats.json").write_text(stats_text, encoding="utf-8")
    identity["stats_sha256"] = hashlib.sha256(stats_text.encode()).hexdigest()
    identity["training_frame_count"] = int(selected_stats["action"]["count"][0])
    identity["normalization_frame_count"] = int(stats["action"]["count"][0])
    manifest_path.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a leakage-safe LeRobot dataset overlay with selected-episode statistics"
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument(
        "--stats-source-root",
        type=Path,
        help="Use normalization statistics from an allowed anchor dataset",
    )
    args = parser.parse_args()
    output = ensure_subset_overlay(
        args.source_root,
        args.output_root,
        args.episodes,
        stats_source_root=args.stats_source_root,
    )
    print(output)


if __name__ == "__main__":
    main()
