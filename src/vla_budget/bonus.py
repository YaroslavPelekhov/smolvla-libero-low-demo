from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .progress_critic import _decode_video, _evaluation_episodes


def _safe_spearman(left: list[float], right: list[float]) -> float | None:
    if len(set(left)) < 2 or len(set(right)) < 2:
        return None
    statistic = float(spearmanr(left, right).statistic)
    return statistic if np.isfinite(statistic) else None


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
        "small_spearman": _safe_spearman(
            truth, [r["small_critic"] for r in records]
        ),
        "robometer_spearman": _safe_spearman(
            truth, [r["robometer"] for r in records]
        ),
        "records": records,
        "policy_optimization_warning": (
            "Neither critic is safe as a direct policy objective without held-out human/simulator audits; "
            "temporal shortcuts and terminal-state poses are exploitable."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(output)


def optimize_selection_payload(
    payload: dict,
    audit_truth: dict[tuple[str, int, int, int], float] | None = None,
    audit_counts: dict[tuple[str, int, int, int], int] | None = None,
) -> dict:
    """Audit zeroth-order policy optimization over the frozen checkpoint bank.

    The reward model selects one of four policies per task/budget. Selection sees
    only the proxy score; the 20-rollout simulator success is used afterwards as
    a held-out audit target. This does not claim gradient-based policy learning.
    """
    grouped: dict[tuple[int, int], list[dict]] = collections.defaultdict(list)
    for record in payload["records"]:
        method, budget, task_id, _ = record["key"]
        if method in {"baseline", "method"}:
            copied = dict(record)
            key = (str(method), int(budget), int(task_id), int(record["key"][3]))
            copied["audit_success"] = (
                float(audit_truth[key]) if audit_truth is not None else float(record["true_success"])
            )
            copied["audit_n"] = int(audit_counts[key]) if audit_counts is not None else 20
            grouped[(int(budget), int(task_id))].append(copied)
    if not grouped or any(len(group) != 4 for group in grouped.values()):
        sizes = {str(key): len(value) for key, value in sorted(grouped.items())}
        raise ValueError(f"Expected four candidate policies in every task/budget group: {sizes}")

    baseline_reference = float(
        np.mean(
            [
                np.mean(
                    [
                        record["audit_success"]
                        for record in group
                        if record["key"][0] == "baseline"
                    ]
                )
                for group in grouped.values()
            ]
        )
    )
    pool_mean = float(
        np.mean(
            [np.mean([record["audit_success"] for record in group]) for group in grouped.values()]
        )
    )
    oracle_mean = float(
        np.mean([max(record["audit_success"] for record in group) for group in grouped.values()])
    )

    selectors = {}
    for score_name in ("small_critic", "robometer"):
        selections = []
        for (budget, task_id), group in sorted(grouped.items()):
            selected = max(
                group,
                key=lambda record: (float(record[score_name]), tuple(record["key"])),
            )
            oracle_success = max(float(record["audit_success"]) for record in group)
            selections.append(
                {
                    "budget": budget,
                    "task_id": task_id,
                    "selected_key": selected["key"],
                    "selected_proxy_score": float(selected[score_name]),
                    "selected_true_success": float(selected["audit_success"]),
                    "audit_episodes": int(selected["audit_n"]),
                    "oracle_true_success": oracle_success,
                    "regret": oracle_success - float(selected["audit_success"]),
                }
            )
        selected_mean = float(np.mean([row["selected_true_success"] for row in selections]))
        selectors[score_name] = {
            "mean_selected_true_success": selected_mean,
            "delta_vs_baseline_reference": selected_mean - baseline_reference,
            "mean_regret_to_oracle": oracle_mean - selected_mean,
            "suboptimal_selections": sum(row["regret"] > 1e-12 for row in selections),
            "selections": selections,
        }

    return {
        "schema_version": 1,
        "optimizer": "argmax proxy score over four frozen policies per task/budget",
        "audit_design": (
            "Five deterministic videos score each proxy and select a checkpoint; simulator success "
            "is then computed on the remaining 15 episodes, which were not proxy-scored."
            if audit_truth is not None
            else "Proxy scores select a checkpoint; simulator success is revealed after selection."
        ),
        "n_task_budget_groups": len(grouped),
        "candidates_per_group": 4,
        "heldout_audit_episodes_per_candidate": (
            min(audit_counts.values()) if audit_counts is not None else 20
        ),
        "baseline_reference_success": baseline_reference,
        "candidate_pool_mean_success": pool_mean,
        "oracle_mean_success": oracle_mean,
        "selectors": selectors,
        "scope": (
            "Finite-bank zeroth-order policy selection, not gradient optimization of SmolVLA. "
            "It is a direct test of whether maximizing each learned video reward improves policy choice."
        ),
    }


def _heldout_success(
    per_run_path: Path, proxy_scores_path: Path
) -> tuple[dict[tuple[str, int, int, int], float], dict[tuple[str, int, int, int], int]]:
    proxy_rows = json.loads(proxy_scores_path.read_text(encoding="utf-8"))
    scored_indices: dict[tuple[str, int, int, int], set[int]] = collections.defaultdict(set)
    for row in proxy_rows:
        key = (str(row["method"]), int(row["budget"]), int(row["task_id"]), int(row["seed"]))
        if key[0] in {"baseline", "method"}:
            scored_indices[key].add(int(row["episode_index"]))

    truth = {}
    counts = {}
    with per_run_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (str(row["method"]), int(row["budget"]), int(row["task_id"]), int(row["seed"]))
            if key[0] not in {"baseline", "method"}:
                continue
            vector = [bool(value) for value in json.loads(row["success_vector"])]
            excluded = scored_indices.get(key, set())
            if len(excluded) != 5:
                raise ValueError(f"Expected five proxy-scored episodes for {key}, got {excluded}")
            heldout = [value for index, value in enumerate(vector) if index not in excluded]
            if len(heldout) != 15:
                raise ValueError(f"Expected fifteen held-out outcomes for {key}")
            truth[key] = float(np.mean(heldout))
            counts[key] = len(heldout)
    expected_keys = {
        (str(row["method"]), int(row["budget"]), int(row["task_id"]), int(row["seed"]))
        for row in proxy_rows
        if row["method"] in {"baseline", "method"}
    }
    if set(truth) != expected_keys:
        raise ValueError("Per-run evidence and proxy-score candidate keys do not match")
    return truth, counts


def optimize_selection(args) -> None:
    payload = json.loads(Path(args.ranking).read_text(encoding="utf-8"))
    truth, counts = _heldout_success(Path(args.per_run), Path(args.proxy_scores))
    result = optimize_selection_payload(payload, truth, counts)
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
    optimize_parser = subparsers.add_parser("optimize-selection")
    optimize_parser.add_argument("--ranking", default="artifacts/results/reward_ranking.json")
    optimize_parser.add_argument("--per-run", default="artifacts/results/per_run.csv")
    optimize_parser.add_argument(
        "--proxy-scores", default="artifacts/results/progress_scores.json"
    )
    optimize_parser.add_argument(
        "--output", default="artifacts/results/reward_optimization.json"
    )
    optimize_parser.set_defaults(function=optimize_selection)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
