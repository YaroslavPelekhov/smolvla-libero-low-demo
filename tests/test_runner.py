from pathlib import Path

from vla_budget.runner import _latest_resume_config, build_runs


def _cfg(tmp_path: Path):
    return {
        "dataset": {
            "repo_id": "test/libero",
            "local_dir": str(tmp_path / "data"),
            "seen_suite": "libero_90",
            "target_suite": "libero_goal",
            "target_task_ids": [0, 1, 2],
        },
        "model": {
            "base": "lerobot/smolvla_base",
            "seen_steps": 100,
            "adaptation_steps": 20,
            "batch_size": 2,
            "n_action_steps": 10,
        },
        "protocol": {
            "budgets": [5, 10, 25],
            "train_seeds": [17, 29],
            "eval_seed": 1000,
            "eval_episodes": 20,
        },
        "method": {"name": "seen_anchored_normalization"},
        "paths": {"outputs": str(tmp_path / "outputs")},
    }


def _manifest():
    return {
        "target": {
            "tasks": {
                str(task): {
                    "prompt": f"task {task}",
                    "wrong_prompt_task_id": (task + 1) % 3,
                    "episodes": {
                        "5": list(range(task * 100, task * 100 + 5)),
                        "10": list(range(task * 100, task * 100 + 10)),
                        "25": list(range(task * 100, task * 100 + 25)),
                    },
                }
                for task in range(3)
            }
        }
    }


def test_method_matrix_is_complete_and_seen_anchored(tmp_path):
    runs = build_runs(_cfg(tmp_path), _manifest(), "method", recording=True)
    assert len(runs) == 36  # 3 budgets x 3 tasks x 2 seeds x (train + eval)
    trains = [run for run in runs if run.kind == "train"]
    assert all(not any(arg.startswith("--peft.") for arg in run.command) for run in trains)
    assert all(run.subset is not None for run in trains)
    assert all(run.subset[3] == tmp_path / "data" / "libero_90" for run in trains)
    assert all(run.metadata["normalization_scope"] == "seen_anchor" for run in trains)
    assert all("--cudnn_deterministic=true" in run.command for run in trains)
    assert all("--dataset.video_backend=pyav" in run.command for run in trains)
    assert all("--policy.input_features=null" in run.command for run in trains)
    assert all("--policy.output_features=null" in run.command for run in trains)
    assert all("--policy.push_to_hub=false" in run.command for run in trains)
    evals = [run for run in runs if run.kind == "eval"]
    assert all("--eval.n_episodes=20" in run.command for run in evals)
    # LeRobot still writes MP4s with this false. Its optional dataset recorder
    # rejects LIBERO's slash-delimited raw feature names in the pinned release.
    assert all("--eval.recording=false" in run.command for run in evals)
    assert all(
        '--env.camera_name_mapping={"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}'
        in run.command
        for run in evals
    )
    assert all("--env.gripper_action_mode=binary_01" in run.command for run in evals)


def test_baseline_has_no_method_flags_and_exact_prefix(tmp_path):
    runs = build_runs(_cfg(tmp_path), _manifest(), "baseline")
    train = next(
        run
        for run in runs
        if run.kind == "train" and run.metadata["budget"] == 5 and run.metadata["task_id"] == 1
    )
    assert "--dataset.episodes=[100,101,102,103,104]" in train.command
    assert not any(argument.startswith("--peft.") for argument in train.command)


def test_stabilized_stage_reuses_two_seed_policies_without_training(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["method"] = {"name": "cross_seed_weight_interpolation", "partner_weight": 0.25}
    runs = build_runs(cfg, _manifest(), "stabilized")
    assert len(runs) == 18
    assert all(run.kind == "eval" for run in runs)
    assert {run.metadata["seed"] for run in runs} == {17, 29}
    assert all(run.metadata["interpolation_partner_weight"] == 0.25 for run in runs)
    assert all("/train/stabilized/" in run.command[1].replace("\\", "/") for run in runs)


def test_latest_resume_config_uses_highest_complete_checkpoint(tmp_path):
    output = tmp_path / "run"
    for step in (10000, 30000, 20000):
        model_dir = output / "checkpoints" / f"{step:06d}" / "pretrained_model"
        model_dir.mkdir(parents=True)
        (model_dir / "train_config.json").write_text("{}", encoding="utf-8")
    (output / "checkpoints" / "040000" / "pretrained_model").mkdir(parents=True)

    result = _latest_resume_config(output)

    assert result == output / "checkpoints" / "030000" / "pretrained_model" / "train_config.json"
