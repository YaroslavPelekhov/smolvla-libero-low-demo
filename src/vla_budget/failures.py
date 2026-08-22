from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


def _sample_video(path: Path):
    import av

    with av.open(str(path)) as container:
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    if not frames:
        raise ValueError(f"No video frames decoded from {path}")
    indices = [0, len(frames) // 2, len(frames) - 1]
    return [(frames[index], index) for index in indices], len(frames)


def _candidate_failures(eval_root: Path, stage: str | None = None):
    candidates = []
    for info_path in sorted(eval_root.rglob("eval_info.json")):
        meta_path = info_path.parent / "run_meta.json"
        if not meta_path.exists():
            continue
        run_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if run_meta.get("stage") == "language":
            continue
        if stage is not None and run_meta.get("stage") != stage:
            continue
        info = json.loads(info_path.read_text(encoding="utf-8"))
        for task in info.get("per_task", []):
            metrics = task["metrics"]
            successes = [bool(value) for value in metrics["successes"]]
            videos = [Path(value) for value in metrics["video_paths"]]
            for fallback_index, video_path in enumerate(videos):
                match = re.search(r"eval_episode_(\d+)$", video_path.stem)
                episode_index = int(match.group(1)) if match else fallback_index
                if episode_index < len(successes) and not successes[episode_index]:
                    candidates.append((run_meta, episode_index, video_path))
    return candidates


def _select_diverse(candidates):
    selected = []
    # Each target task is represented. Budgets are staggered to avoid three
    # nearly identical examples from the first lexicographic run directory.
    desired_budgets = {0: 5, 1: 10, 2: 25}
    for task_id, desired_budget in desired_budgets.items():
        task_candidates = [item for item in candidates if int(item[0]["task_id"]) == task_id]
        task_candidates.sort(
            key=lambda item: (
                0 if item[0]["method"] == "method" else 1,
                abs(int(item[0]["budget"]) - desired_budget),
                int(item[0]["seed"]),
                item[1],
            )
        )
        if task_candidates:
            selected.append(task_candidates[0])
    # Some evaluators cap videos below the registered episode count, so a task
    # can have measured failures without a recorded failed rollout. Fill from
    # distinct run cells rather than fabricating or relabeling a success video.
    if len(selected) < 3:
        used = {(item[0]["budget"], item[0]["task_id"], item[0]["seed"]) for item in selected}
        for item in sorted(
            candidates,
            key=lambda value: (
                int(value[0]["task_id"]),
                int(value[0]["budget"]),
                int(value[0]["seed"]),
                value[1],
            ),
        ):
            cell = (item[0]["budget"], item[0]["task_id"], item[0]["seed"])
            if cell not in used:
                selected.append(item)
                used.add(cell)
            if len(selected) == 3:
                break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract three reproducible failed rollout contact sheets")
    parser.add_argument("--eval-root", default="outputs/eval")
    parser.add_argument("--output", default="artifacts/results/failures")
    parser.add_argument("--stage", help="Restrict failures to one evaluation stage")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    selected = _select_diverse(_candidate_failures(Path(args.eval_root), args.stage))
    if len(selected) < 3:
        raise SystemExit(
            f"Only {len(selected)} recorded failure cells are available; need three"
        )

    cases = []
    for case_id, (run_meta, episode_index, video_path) in enumerate(selected, start=1):
        sampled, frame_count = _sample_video(video_path)
        figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.2))
        for axis, (image, frame_index), label in zip(
            axes, sampled, ["start", "middle", "end"], strict=True
        ):
            axis.imshow(image)
            axis.set_title(f"{label} · frame {frame_index}", fontsize=8)
            axis.axis("off")
        figure.suptitle(
            f"F{case_id} · {run_meta['method']} · {run_meta['budget']} demos · "
            f"task {run_meta['task_id']} · seed {run_meta['seed']} · episode {episode_index}",
            fontsize=8.5,
        )
        figure.tight_layout(pad=0.3)
        figure_path = output / f"failure-{case_id}.png"
        figure.savefig(figure_path, dpi=180)
        plt.close(figure)
        cases.append(
            {
                **run_meta,
                "episode_index": episode_index,
                "video": str(video_path),
                "video_frames": frame_count,
                "figure": str(figure_path),
            }
        )
    (output / "catalog.json").write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
