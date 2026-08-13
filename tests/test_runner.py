from pathlib import Path

from vla_budget.runner import build_runs


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
        "method": {"lora_rank": 16, "max_num_transforms": 2},
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


def test_method_matrix_is_complete_and_regularized(tmp_path):
    runs = build_runs(_cfg(tmp_path), _manifest(), "method", recording=True)
    assert len(runs) == 36  # 3 budgets x 3 tasks x 2 seeds x (train + eval)
    trains = [run for run in runs if run.kind == "train"]
    assert all("--peft.method_type=LORA" in run.command for run in trains)
    assert all("--peft.lora_alpha=16" in run.command for run in trains)
    assert all("--dataset.image_transforms.enable=true" in run.command for run in trains)
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


def test_baseline_has_no_method_flags_and_exact_prefix(tmp_path):
    runs = build_runs(_cfg(tmp_path), _manifest(), "baseline")
    train = next(
        run
        for run in runs
        if run.kind == "train" and run.metadata["budget"] == 5 and run.metadata["task_id"] == 1
    )
    assert "--dataset.episodes=[100,101,102,103,104]" in train.command
    assert not any(argument.startswith("--peft.") for argument in train.command)
