from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .config import load_config, resolve
from .protocol import manifest_digest


@dataclass(frozen=True)
class Run:
    kind: str
    output_dir: Path
    command: list[str]
    metadata: dict


def _json_list(values: Iterable[int]) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"


def _entrypoint(name: str) -> str:
    candidate = Path(sys.executable).with_name(name + (".exe" if os.name == "nt" else ""))
    return str(candidate) if candidate.exists() else name


def _checkpoint(train_dir: Path) -> Path:
    preferred = train_dir / "checkpoints" / "last" / "pretrained_model"
    if preferred.exists():
        return preferred
    checkpoints = sorted((train_dir / "checkpoints").glob("*/pretrained_model"))
    return checkpoints[-1] if checkpoints else preferred


def _common_train(cfg: dict, policy: Path | str, root: Path, output: Path, seed: int) -> list[str]:
    model = cfg["model"]
    data = cfg["dataset"]
    command = [
        _entrypoint("lerobot-train"),
        f"--policy.path={policy}",
        "--policy.repo_id=null",
        "--policy.push_to_hub=false",
        # smolvla_base was trained on a different robot schema (3 cameras, 6-D
        # state/action). Nulling the serialized feature map makes LeRobot infer the
        # two-camera, 8-D state, 7-D action LIBERO schema while retaining weights.
        "--policy.input_features=null",
        "--policy.output_features=null",
        f"--policy.n_action_steps={model['n_action_steps']}",
        f"--dataset.repo_id={data['repo_id']}",
        f"--dataset.root={root}",
        f"--dataset.revision={data.get('revision', 'main')}",
        "--dataset.video_backend=pyav",
        "--dataset.return_uint8=true",
        f"--output_dir={output}",
        f"--batch_size={model['batch_size']}",
        f"--num_workers={model.get('num_workers', 4)}",
        f"--seed={seed}",
        "--cudnn_deterministic=true",
        f"--accelerator.mixed_precision={model.get('mixed_precision', 'bf16')}",
        "--env_eval_freq=0",
        "--save_freq=0",
        "--wandb.enable=false",
    ]
    if "load_vlm_weights" in model:
        command.append(
            f"--policy.load_vlm_weights={str(model['load_vlm_weights']).lower()}"
        )
    if model.get("vlm_model_name"):
        command.append(f"--policy.vlm_model_name={model['vlm_model_name']}")
    return command


def _eval_command(
    cfg: dict,
    policy: Path | str,
    output: Path,
    task_id: int,
    *,
    recording: bool,
) -> list[str]:
    protocol = cfg["protocol"]
    model = cfg["model"]
    eval_batch_size = int(protocol.get("eval_batch_size", 1))
    return [
        _entrypoint("lerobot-eval"),
        f"--policy.path={policy}",
        f"--policy.n_action_steps={model['n_action_steps']}",
        "--env.type=libero",
        f"--env.task={cfg['dataset']['target_suite']}",
        f"--env.task_ids=[{task_id}]",
        '--env.camera_name_mapping={"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}',
        "--env.control_mode=relative",
        "--env.observation_height=256",
        "--env.observation_width=256",
        f"--env.max_parallel_tasks={eval_batch_size}",
        f"--eval.n_episodes={protocol['eval_episodes']}",
        f"--eval.batch_size={eval_batch_size}",
        f"--eval.use_async_envs={str(protocol.get('eval_async', False)).lower()}",
        # LeRobot 0.6.2 always writes an MP4. Its optional rollout-dataset
        # recorder rejects LIBERO's slash-delimited raw feature names, so keep
        # that separate path disabled; failure analysis reads the MP4s.
        "--eval.recording=false",
        f"--seed={protocol['eval_seed']}",
        f"--output_dir={output}",
    ]


def build_runs(cfg: dict, manifest: dict, stage: str, recording: bool = False) -> list[Run]:
    output_root = resolve(cfg["paths"]["outputs"])
    data_root = resolve(cfg["dataset"]["local_dir"])
    seen_root = data_root / cfg["dataset"]["seen_suite"]
    target_root = data_root / cfg["dataset"]["target_suite"]
    seeds = [int(value) for value in cfg["protocol"]["train_seeds"]]
    budgets = [int(value) for value in cfg["protocol"]["budgets"]]
    task_ids = [int(value) for value in cfg["dataset"]["target_task_ids"]]
    runs: list[Run] = []

    if stage == "seen":
        for seed in seeds:
            output = output_root / "train" / "seen" / f"seed-{seed}"
            command = _common_train(cfg, cfg["model"]["base"], seen_root, output, seed)
            command.append(f"--steps={cfg['model']['seen_steps']}")
            runs.append(Run("train", output, command, {"stage": "seen", "seed": seed}))
        return runs

    if stage in {"zero", "language"}:
        for seed in seeds:
            policy = _checkpoint(output_root / "train" / "seen" / f"seed-{seed}")
            for task_id in task_ids:
                label = "correct" if stage == "zero" else "wrong"
                output = output_root / "eval" / stage / f"task-{task_id}" / f"seed-{seed}"
                command = _eval_command(cfg, policy, output, task_id, recording=recording)
                if stage == "language":
                    wrong_id = int(manifest["target"]["tasks"][str(task_id)]["wrong_prompt_task_id"])
                    wrong_prompt = manifest["target"]["tasks"][str(wrong_id)]["prompt"]
                    command = [
                        sys.executable,
                        "-m",
                        "vla_budget.wrong_prompt_eval",
                        f"--wrong-prompt={wrong_prompt}",
                        "--",
                        *command[1:],
                    ]
                meta = {
                    "stage": stage,
                    "method": label,
                    "budget": 0,
                    "task_id": task_id,
                    "seed": seed,
                }
                runs.append(Run("eval", output, command, meta))
        return runs

    if stage not in {"baseline", "method"}:
        raise ValueError(stage)

    for budget in budgets:
        for task_id in task_ids:
            episodes = manifest["target"]["tasks"][str(task_id)]["episodes"][str(budget)]
            for seed in seeds:
                seen_policy = _checkpoint(output_root / "train" / "seen" / f"seed-{seed}")
                train_dir = (
                    output_root
                    / "train"
                    / stage
                    / f"budget-{budget}"
                    / f"task-{task_id}"
                    / f"seed-{seed}"
                )
                command = _common_train(cfg, seen_policy, target_root, train_dir, seed)
                command.extend(
                    [
                        f"--dataset.episodes={_json_list(episodes)}",
                        f"--steps={cfg['model']['adaptation_steps']}",
                    ]
                )
                if stage == "method":
                    command.extend(
                        [
                            "--peft.method_type=LORA",
                            f"--peft.r={cfg['method']['lora_rank']}",
                            f"--peft.lora_alpha={cfg['method'].get('lora_alpha', cfg['method']['lora_rank'])}",
                            "--dataset.image_transforms.enable=true",
                            "--dataset.image_transforms.random_order=true",
                            f"--dataset.image_transforms.max_num_transforms={cfg['method']['max_num_transforms']}",
                        ]
                    )
                train_meta = {
                    "stage": stage,
                    "method": stage,
                    "budget": budget,
                    "task_id": task_id,
                    "seed": seed,
                    "episodes": episodes,
                }
                runs.append(Run("train", train_dir, command, train_meta))

                eval_dir = (
                    output_root
                    / "eval"
                    / stage
                    / f"budget-{budget}"
                    / f"task-{task_id}"
                    / f"seed-{seed}"
                )
                eval_command = _eval_command(
                    cfg, _checkpoint(train_dir), eval_dir, task_id, recording=recording
                )
                runs.append(Run("eval", eval_dir, eval_command, train_meta))
    return runs


def _format(command: list[str]) -> str:
    return shlex.join(command)


def _execute_run(run: Run, *, stream_output: bool) -> None:
    if run.kind == "eval":
        policy_arg = next(arg for arg in run.command if arg.startswith("--policy.path="))
        policy_value = policy_arg.split("=", 1)[1]
        policy_path = Path(policy_value)
        is_hub_id = "/" in policy_value and not policy_path.is_absolute()
        if not is_hub_id and not policy_path.exists():
            raise FileNotFoundError(f"Policy checkpoint not found: {policy_path}")
    if run.output_dir.exists():
        raise FileExistsError(
            f"Output exists without a COMPLETED marker: {run.output_dir}. "
            "Move it aside or choose a new --output-root."
        )
    run.output_dir.parent.mkdir(parents=True, exist_ok=True)
    if stream_output:
        subprocess.run(run.command, check=True)
    else:
        temp_log = run.output_dir.parent / f".{run.output_dir.name}.run.log"
        with temp_log.open("w", encoding="utf-8") as log:
            subprocess.run(run.command, check=True, stdout=log, stderr=subprocess.STDOUT)
        temp_log.replace(run.output_dir / "run.log")
    (run.output_dir / "run_meta.json").write_text(
        json.dumps(run.metadata, indent=2) + "\n", encoding="utf-8"
    )
    (run.output_dir / "command.txt").write_text(_format(run.command) + "\n", encoding="utf-8")
    (run.output_dir / "COMPLETED").write_text("ok\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the preregistered LIBERO experiment matrix")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--stage", required=True, choices=["seen", "zero", "language", "baseline", "method"]
    )
    parser.add_argument("--execute", action="store_true", help="Run commands; default is a dry run")
    parser.add_argument(
        "--recording",
        action="store_true",
        help="Deprecated compatibility flag; MP4 rollouts are written automatically",
    )
    parser.add_argument(
        "--data-root",
        help="Override dataset.local_dir (recommended: native Linux storage under WSL)",
    )
    parser.add_argument("--output-root", help="Override paths.outputs")
    parser.add_argument("--base-model", help="Override model.base with a local checkpoint or Hub id")
    parser.add_argument(
        "--vlm-model",
        help="Override model.vlm_model_name (use a local config/processor cache offline)",
    )
    parser.add_argument(
        "--limit-runs",
        type=int,
        help="Execute or print only the first N run units for diagnostic runs",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip run units that already contain a COMPLETED marker",
    )
    parser.add_argument("--kind", choices=["all", "train", "eval"], default="all")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Concurrent run units (use only for evaluation unless GPU capacity is known)",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.data_root:
        cfg["dataset"]["local_dir"] = args.data_root
    if args.output_root:
        cfg["paths"]["outputs"] = args.output_root
    if args.base_model:
        cfg["model"]["base"] = args.base_model
    if args.vlm_model:
        cfg["model"]["vlm_model_name"] = args.vlm_model
    manifest_path = resolve(cfg["paths"]["manifest"])
    prereg_path = resolve(cfg["paths"]["preregistration"])
    if not manifest_path.exists() or not prereg_path.exists():
        raise SystemExit("Run vla-download, vla-manifest and vla-preregister first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("sha256") != manifest_digest(manifest):
        raise SystemExit("Protocol manifest hash mismatch")
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    prediction_hash = hashlib.sha256(resolve("predictions.yaml").read_bytes()).hexdigest()
    if prereg.get("sha256") != prediction_hash:
        raise SystemExit("Predictions changed after preregistration")
    runs = build_runs(cfg, manifest, args.stage, recording=args.recording)
    if args.kind != "all":
        runs = [run for run in runs if run.kind == args.kind]
    if args.limit_runs is not None:
        if args.limit_runs <= 0:
            raise SystemExit("--limit-runs must be positive")
        runs = runs[: args.limit_runs]
    selected: list[Run] = []
    for index, run in enumerate(runs, start=1):
        print(f"[{index}/{len(runs)}] {run.kind}: {_format(run.command)}", flush=True)
        if not args.execute:
            continue
        if args.skip_completed and (run.output_dir / "COMPLETED").exists():
            print(f"  skip: {run.output_dir} is already complete", flush=True)
            continue
        selected.append(run)
    if not args.execute or not selected:
        return
    if os.name == "nt":
        raise SystemExit("LIBERO execution requires Linux; dry-run is supported on Windows")
    if args.jobs <= 0:
        raise SystemExit("--jobs must be positive")
    if args.jobs > 1 and any(run.kind == "train" for run in selected):
        raise SystemExit("Concurrent training is disabled; use --jobs=1 for train units")
    if args.jobs == 1:
        for run in selected:
            _execute_run(run, stream_output=True)
        return
    failures = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        future_map = {executor.submit(_execute_run, run, stream_output=False): run for run in selected}
        for future in as_completed(future_map):
            run = future_map[future]
            try:
                future.result()
                print(f"complete: {run.output_dir}", flush=True)
            except Exception as error:  # noqa: BLE001 - collect all independent run failures
                failures.append((run.output_dir, error))
                print(f"failed: {run.output_dir}: {error}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} concurrent run units failed")


if __name__ == "__main__":
    main()
