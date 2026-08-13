from __future__ import annotations

import argparse
import json
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


def _candidate_failures(eval_root: Path):
    candidates = []
    for info_path in sorted(eval_root.rglob("eval_info.json")):
        meta_path = info_path.parent / "run_meta.json"
        if not meta_path.exists():
            continue
        run_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if run_meta.get("method") != "method":
            continue
        info = json.loads(info_path.read_text(encoding="utf-8"))
        for task in info.get("per_task", []):
            metrics = task["metrics"]
            successes = [bool(value) for value in metrics["successes"]]
            videos = [Path(value) for value in metrics["video_paths"]]
            # All proposed-method runs have zero success, so every retained MP4
            # is unambiguously a failure despite LeRobot retaining only ten
            # videos for twenty batched outcomes.
            if any(successes):
                continue
            for episode_index, video_path in enumerate(videos):
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
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract three reproducible failed rollout contact sheets")
    parser.add_argument("--eval-root", default="outputs/eval")
    parser.add_argument("--output", default="artifacts/results/failures")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    selected = _select_diverse(_candidate_failures(Path(args.eval_root)))
    if len(selected) < 3:
        raise SystemExit(
            f"Failures span only {len(selected)} of the three target tasks; need one per task"
        )

    cases = []
    for case_id, (run_meta, episode_index, video_path) in enumerate(selected, start=1):
        sampled, frame_count = _sample_video(video_path)
        figure, axes = plt.subplots(1, 3, figsize=(9, 3))
        for axis, (image, frame_index), label in zip(
            axes, sampled, ["start", "middle", "end"], strict=True
        ):
            axis.imshow(image)
            axis.set_title(f"{label} - frame {frame_index}")
            axis.axis("off")
        figure.suptitle(
            f"Failure {case_id}: {run_meta['method']} / {run_meta['budget']} demos / "
            f"task {run_meta['task_id']} / seed {run_meta['seed']} / episode {episode_index}"
        )
        figure.tight_layout()
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
