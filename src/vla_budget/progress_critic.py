from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from .metadata import load_suite_metadata


def _torch_modules():
    import torch
    from torch import nn

    return torch, nn


def make_model():
    _, nn = _torch_modules()
    return nn.Sequential(
        nn.Conv2d(3, 16, 5, stride=2, padding=2),
        nn.SiLU(),
        nn.Conv2d(16, 32, 3, stride=2, padding=1),
        nn.SiLU(),
        nn.Conv2d(32, 64, 3, stride=2, padding=1),
        nn.SiLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    )


class ProgressFrames:
    """Frame labels use only temporal position; action values are never accessed."""

    def __init__(self, dataset, episode_rows: list[dict], image_key: str, stride: int = 8):
        self.dataset = dataset
        self.image_key = image_key
        self.items: list[tuple[int, float]] = []
        for row in episode_rows:
            start, end = int(row["dataset_from_index"]), int(row["dataset_to_index"])
            length = max(1, end - start)
            for offset in range(0, length, stride):
                self.items.append((start + offset, offset / max(1, length - 1)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        torch, _ = _torch_modules()
        frame_index, target = self.items[index]
        image = self.dataset[frame_index][self.image_key]
        image = image.float()
        if image.max() > 1:
            image = image / 255.0
        return image, torch.tensor([target], dtype=torch.float32)


def _episode_rows(root: Path) -> list[dict]:
    import pyarrow.parquet as pq

    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    return pq.read_table(files, columns=["episode_index", "dataset_from_index", "dataset_to_index"]).to_pylist()


def train(args) -> None:
    torch, _ = _torch_modules()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from torch.utils.data import DataLoader

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = Path(args.dataset_root)
    metadata = load_suite_metadata(root)
    if metadata.total_tasks < 50:
        raise ValueError("The progress critic must be trained on the seen LIBERO-90 pool")
    rows = _episode_rows(root)
    random.Random(args.seed).shuffle(rows)
    split = max(1, int(0.9 * len(rows)))
    train_rows, val_rows = rows[:split], rows[split:]
    dataset = LeRobotDataset(args.repo_id, root=root)
    training = ProgressFrames(dataset, train_rows, args.image_key, args.stride)
    validation = ProgressFrames(dataset, val_rows, args.image_key, args.stride)
    train_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(validation, batch_size=args.batch_size, num_workers=4)

    device = torch.device(args.device)
    model = make_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = torch.nn.HuberLoss()
    history = []
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for images, targets in train_loader:
            loss = loss_fn(model(images.to(device)), targets.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        val_losses = []
        with torch.no_grad():
            for images, targets in val_loader:
                val_losses.append(loss_fn(model(images.to(device)), targets.to(device)).item())
        record = {
            "epoch": epoch + 1,
            "train_loss": sum(train_losses) / len(train_losses),
            "val_loss": sum(val_losses) / max(1, len(val_losses)),
        }
        history.append(record)
        print(record)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "image_key": args.image_key,
            "history": history,
            "data_statement": "LIBERO seen frames only; labels are normalized timestamps; actions unused",
        },
        output,
    )


def _evaluation_episodes(eval_root: str | Path, max_per_run: int | None = None):
    """Yield the same deterministic rollout subset for both reward-free critics."""
    for info_path in sorted(Path(eval_root).rglob("eval_info.json")):
        run_meta_path = info_path.parent / "run_meta.json"
        if not run_meta_path.exists():
            continue
        run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
        if run_meta.get("stage") == "language":
            continue
        payload = json.loads(info_path.read_text(encoding="utf-8"))
        candidates = []
        for task_group in payload["per_task"]:
            metrics = task_group["metrics"]
            run_success_rate = float(np.mean([bool(value) for value in metrics["successes"]]))
            # LeRobot 0.6.2 reports all 20 outcomes but retains only the ten MP4s
            # from its batched writer. Reward-model scoring therefore uses a
            # deterministic subset of available videos and compares it with the
            # complete 20-episode checkpoint success rate.
            for episode_index, video_path in enumerate(metrics["video_paths"]):
                candidates.append((episode_index, run_success_rate, Path(video_path)))
        if max_per_run and len(candidates) > max_per_run:
            selected = np.linspace(0, len(candidates) - 1, max_per_run).round().astype(int)
            candidates = [candidates[index] for index in selected]
        for episode_index, success_rate, video_path in candidates:
            yield run_meta, episode_index, success_rate, video_path


def _decode_video(path: Path, stride: int = 1) -> list[np.ndarray]:
    import av

    with av.open(str(path)) as container:
        return [
            frame.to_ndarray(format="rgb24")
            for index, frame in enumerate(container.decode(video=0))
            if index % stride == 0
        ]


def score(args) -> None:
    torch, _ = _torch_modules()

    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = make_model().to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    rows = []
    episodes = _evaluation_episodes(args.eval_root, args.max_episodes_per_run)
    for run_meta, episode_index, true_success_rate, video_path in episodes:
        frames = _decode_video(video_path, max(1, args.stride))
        if not frames:
            raise ValueError(f"No frames decoded from {video_path}")
        images = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0
        with torch.no_grad():
            progress = model(images.to(device)).squeeze(1).cpu().numpy()
        rows.append(
            {
                **run_meta,
                "episode_index": episode_index,
                "critic_score": float(progress[-max(1, len(progress) // 10) :].mean()),
                "true_success_rate": true_success_rate,
                "video_path": str(video_path),
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Action-free temporal progress critic")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--dataset-root", required=True)
    train_parser.add_argument("--repo-id", default="nvidia/LIBERO_LeRobot_v3")
    train_parser.add_argument("--image-key", default="observation.images.image")
    train_parser.add_argument("--output", default="artifacts/progress_critic.pt")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--stride", type=int, default=8)
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--seed", type=int, default=17)
    train_parser.add_argument("--device", default="cuda")
    train_parser.set_defaults(function=train)
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--checkpoint", required=True)
    score_parser.add_argument("--eval-root", default="outputs/eval")
    score_parser.add_argument("--output", default="artifacts/results/progress_scores.json")
    score_parser.add_argument("--stride", type=int, default=4)
    score_parser.add_argument("--max-episodes-per-run", type=int, default=5)
    score_parser.add_argument("--device", default="cuda")
    score_parser.set_defaults(function=score)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
