from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq


@dataclass(frozen=True)
class SuiteMetadata:
    root: Path
    total_episodes: int
    total_tasks: int
    task_prompts: dict[int, str]
    episodes_by_task: dict[int, tuple[int, ...]]


def _read_parquet_tree(path: Path):
    files = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No parquet metadata below {path}")
    return pq.read_table(files)


def _prompt_column(column_names: list[str]) -> str:
    candidates = [
        "task",
        "prompt",
        "language_instruction",
        "__index_level_0__",
    ]
    for candidate in candidates:
        if candidate in column_names:
            return candidate
    raise ValueError(f"Cannot identify task prompt column in {column_names}")


def load_suite_metadata(root: str | Path) -> SuiteMetadata:
    root = Path(root)
    info_path = root / "meta" / "info.json"
    with info_path.open(encoding="utf-8") as stream:
        info = json.load(stream)

    tasks_table = _read_parquet_tree(root / "meta" / "tasks.parquet")
    prompt_col = _prompt_column(tasks_table.column_names)
    task_prompts = {
        int(row["task_index"]): str(row[prompt_col]) for row in tasks_table.to_pylist()
    }

    episodes_table = _read_parquet_tree(root / "meta" / "episodes")
    required = {"episode_index", "tasks"}
    if not required.issubset(episodes_table.column_names):
        raise ValueError(
            f"Episode metadata must contain {sorted(required)}; got {episodes_table.column_names}"
        )
    prompt_to_task = {prompt: task_id for task_id, prompt in task_prompts.items()}
    grouped: dict[int, list[int]] = {task_id: [] for task_id in task_prompts}
    for row in episodes_table.select(["episode_index", "tasks"]).to_pylist():
        prompts = row["tasks"] or []
        if len(prompts) != 1:
            raise ValueError(
                f"Episode {row['episode_index']} has {len(prompts)} task labels; expected exactly one"
            )
        prompt = str(prompts[0])
        if prompt not in prompt_to_task:
            raise ValueError(f"Unknown task prompt in episode metadata: {prompt!r}")
        grouped[prompt_to_task[prompt]].append(int(row["episode_index"]))

    return SuiteMetadata(
        root=root,
        total_episodes=int(info["total_episodes"]),
        total_tasks=int(info["total_tasks"]),
        task_prompts=task_prompts,
        episodes_by_task={key: tuple(sorted(value)) for key, value in grouped.items()},
    )

