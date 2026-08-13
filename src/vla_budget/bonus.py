from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .progress_critic import _decode_video, _evaluation_episodes


def robometer_video(args) -> None:
    import torch
    from lerobot.rewards.robometer.configuration_robometer import RobometerConfig
    from lerobot.rewards.robometer.modeling_robometer import RobometerRewardModel
    from lerobot.rewards.robometer.processor_robometer import (
        ROBOMETER_FEATURE_PREFIX,
        RobometerEncoderProcessorStep,
    )

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    saved = json.loads((Path(args.model_path) / "config.json").read_text(encoding="utf-8"))
    config = RobometerConfig(
        pretrained_path=args.model_path,
        base_model_id=args.processor_path,
        device=args.device,
        image_key=saved["image_key"],
        task_key=saved["task_key"],
        default_task=saved["default_task"],
        max_frames=saved["max_frames"],
        reward_output=saved["reward_output"],
        success_threshold=saved["success_threshold"],
        torch_dtype=saved["torch_dtype"],
        use_multi_image=saved["use_multi_image"],
        use_per_frame_progress_token=saved["use_per_frame_progress_token"],
        average_temporal_patches=saved["average_temporal_patches"],
        frame_pooling=saved["frame_pooling"],
        frame_pooling_attn_temperature=saved["frame_pooling_attn_temperature"],
        progress_loss_type=saved["progress_loss_type"],
        progress_discrete_bins=saved["progress_discrete_bins"],
        vlm_config=saved["vlm_config"],
    )
    model = RobometerRewardModel.from_pretrained(args.model_path, config=config)
    model.to(args.device).eval()
    encoder = RobometerEncoderProcessorStep(
        base_model_id=args.processor_path,
        image_key=config.image_key,
        task_key=config.task_key,
        max_frames=args.frames,
        use_multi_image=config.use_multi_image,
        use_per_frame_progress_token=config.use_per_frame_progress_token,
    )

    episodes = list(_evaluation_episodes(args.eval_root, args.max_episodes_per_run))
    rows = []
    for start in range(0, len(episodes), args.batch_size):
        group = episodes[start : start + args.batch_size]
        samples = []
        for meta, _, _, video_path in group:
            video = _decode_video(video_path)
            if not video:
                raise ValueError(f"No frames decoded from {video_path}")
            indices = np.linspace(0, len(video) - 1, args.frames).round().astype(int)
            frames = np.stack([video[index] for index in indices])
            task = manifest["target"]["tasks"][str(meta["task_id"])]["prompt"]
            samples.append((frames, task))
        encoded = encoder.encode_samples(samples)
        batch = {f"{ROBOMETER_FEATURE_PREFIX}{key}": value for key, value in encoded.items()}
        with torch.inference_mode():
            scores = model.compute_reward(batch).float().cpu().tolist()
        for (meta, episode_index, true_success_rate, video_path), score_value in zip(
            group, scores, strict=True
        ):
            rows.append(
                {
                    **meta,
                    "episode_index": episode_index,
                    "robometer_score": float(score_value),
                    "true_success_rate": true_success_rate,
                    "video_path": str(video_path),
                }
            )
        print(f"Robometer: {min(start + len(group), len(episodes))}/{len(episodes)}", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(output)


def compare(args) -> None:
    small_rows = json.loads(Path(args.small_scores).read_text(encoding="utf-8"))
    robot_rows = json.loads(Path(args.robometer_scores).read_text(encoding="utf-8"))
    small_by_run: dict[tuple, list[float]] = {}
    true_by_run: dict[tuple, list[bool]] = {}
    for row in small_rows:
        key = (row["method"], row["budget"], row["task_id"], row["seed"])
        small_by_run.setdefault(key, []).append(float(row["critic_score"]))
        true_by_run.setdefault(key, []).append(float(row["true_success_rate"]))

    robot_by_run: dict[tuple, list[float]] = {}
    for row in robot_rows:
        key = (row["method"], row["budget"], row["task_id"], row["seed"])
        robot_by_run.setdefault(key, []).append(float(row["robometer_score"]))

    records = []
    for key in sorted(set(small_by_run) & set(robot_by_run)):
        records.append(
            {
                "key": key,
                "true_success": float(np.mean(true_by_run[key])),
                "small_critic": float(np.mean(small_by_run[key])),
                "robometer": float(np.mean(robot_by_run[key])),
            }
        )
    if len(records) < 3:
        raise SystemExit("Need at least three checkpoint runs scored by both critics")
    truth = [record["true_success"] for record in records]
    result = {
        "n_checkpoints": len(records),
        "small_spearman": float(spearmanr(truth, [r["small_critic"] for r in records]).statistic),
        "robometer_spearman": float(spearmanr(truth, [r["robometer"] for r in records]).statistic),
        "records": records,
        "policy_optimization_warning": (
            "Neither critic is safe as a direct policy objective without held-out human/simulator audits; "
            "temporal shortcuts and terminal-state poses are exploitable."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bonus B: rollout-video reward-model comparison")
    subparsers = parser.add_subparsers(dest="command", required=True)
    score_parser = subparsers.add_parser("robometer-video")
    score_parser.add_argument("--eval-root", required=True)
    score_parser.add_argument("--model-path", required=True)
    score_parser.add_argument("--processor-path", required=True)
    score_parser.add_argument("--manifest", default="artifacts/protocol_manifest.json")
    score_parser.add_argument("--output", default="artifacts/results/robometer_scores.json")
    score_parser.add_argument("--frames", type=int, default=4)
    score_parser.add_argument("--batch-size", type=int, default=1)
    score_parser.add_argument("--max-episodes-per-run", type=int, default=5)
    score_parser.add_argument("--device", default="cuda")
    score_parser.set_defaults(function=robometer_video)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--small-scores", default="artifacts/results/progress_scores.json")
    compare_parser.add_argument(
        "--robometer-scores", default="artifacts/results/robometer_scores.json"
    )
    compare_parser.add_argument("--output", default="artifacts/results/reward_ranking.json")
    compare_parser.set_defaults(function=compare)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
