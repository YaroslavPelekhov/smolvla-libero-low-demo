from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint(train_dir: Path) -> Path:
    preferred = train_dir / "checkpoints" / "last" / "pretrained_model"
    if preferred.is_dir():
        return preferred
    candidates = sorted((train_dir / "checkpoints").glob("*/pretrained_model"))
    if not candidates:
        raise FileNotFoundError(f"checkpoint not found under {train_dir}")
    return candidates[-1]


def build_task_vector_checkpoint(
    *,
    seen_anchor: Path,
    target_anchor: Path,
    seen_partner: Path,
    target_partner: Path,
    output: Path,
    partner_weight: float,
) -> dict:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    if not 0.0 <= partner_weight <= 1.0:
        raise ValueError("partner_weight must be in [0, 1]")
    sources = {
        "seen_anchor": seen_anchor / "model.safetensors",
        "target_anchor": target_anchor / "model.safetensors",
        "seen_partner": seen_partner / "model.safetensors",
        "target_partner": target_partner / "model.safetensors",
    }
    for label, path in sources.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")

    handles = {
        label: safe_open(path, framework="pt", device="cpu") for label, path in sources.items()
    }
    keys = set(handles["target_anchor"].keys())
    if any(set(handle.keys()) != keys for handle in handles.values()):
        raise ValueError("task-vector source tensor keys differ")

    tensors = {}
    floating = elements = 0
    alpha = float(partner_weight)
    for key in sorted(keys):
        seen_a = handles["seen_anchor"].get_tensor(key)
        target_a = handles["target_anchor"].get_tensor(key)
        seen_b = handles["seen_partner"].get_tensor(key)
        target_b = handles["target_partner"].get_tensor(key)
        shapes = {tensor.shape for tensor in (seen_a, target_a, seen_b, target_b)}
        dtypes = {tensor.dtype for tensor in (seen_a, target_a, seen_b, target_b)}
        if len(shapes) != 1 or len(dtypes) != 1:
            raise ValueError(f"incompatible tensor across sources: {key}")
        if target_a.is_floating_point():
            work_dtype = torch.float32
            delta_a = target_a.to(work_dtype) - seen_a.to(work_dtype)
            delta_b = target_b.to(work_dtype) - seen_b.to(work_dtype)
            mixed = seen_a.to(work_dtype) + (1.0 - alpha) * delta_a + alpha * delta_b
            tensors[key] = mixed.to(target_a.dtype).contiguous()
            floating += 1
            elements += target_a.numel()
        else:
            tensors[key] = target_a.contiguous()

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temp_name:
        temporary = Path(temp_name)
        shutil.copytree(target_anchor, temporary / "checkpoint", dirs_exist_ok=True)
        temporary = temporary / "checkpoint"
        save_file(tensors, temporary / "model.safetensors", metadata={"format": "pt"})
        manifest = {
            "schema_version": 1,
            "method": "cross_seed_task_vector_interpolation",
            "formula": "seen_anchor + (1-alpha)*(target_anchor-seen_anchor) + alpha*(target_partner-seen_partner)",
            "partner_weight": alpha,
            "sources": {
                label: {"path": str(path.parent), "sha256": _sha256(path)}
                for label, path in sources.items()
            },
            "output_sha256": _sha256(temporary / "model.safetensors"),
            "floating_tensors": floating,
            "elements": elements,
            "processor_source": str(target_anchor),
        }
        (temporary / "task_vector_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        if output.exists():
            existing_manifest = output / "task_vector_manifest.json"
            if existing_manifest.is_file() and json.loads(
                existing_manifest.read_text(encoding="utf-8")
            ) == manifest:
                return manifest
            raise FileExistsError(f"refusing to replace existing checkpoint: {output}")
        temporary.replace(output)
    return manifest


def build_matrix(
    output_root: Path,
    budgets: list[int],
    tasks: list[int],
    seeds: list[int],
    alpha: float,
    skip_completed: bool = False,
) -> None:
    if len(seeds) != 2:
        raise ValueError("task-vector interpolation requires exactly two seed branches")
    for budget in budgets:
        for task in tasks:
            for anchor_seed, partner_seed in (seeds, list(reversed(seeds))):
                seen_anchor = _checkpoint(output_root / "train" / "seen" / f"seed-{anchor_seed}")
                seen_partner = _checkpoint(output_root / "train" / "seen" / f"seed-{partner_seed}")
                target_anchor = _checkpoint(
                    output_root
                    / "train"
                    / "baseline"
                    / f"budget-{budget}"
                    / f"task-{task}"
                    / f"seed-{anchor_seed}"
                )
                target_partner = _checkpoint(
                    output_root
                    / "train"
                    / "baseline"
                    / f"budget-{budget}"
                    / f"task-{task}"
                    / f"seed-{partner_seed}"
                )
                destination = (
                    output_root
                    / "train"
                    / "task_vector"
                    / f"budget-{budget}"
                    / f"task-{task}"
                    / f"seed-{anchor_seed}"
                    / "checkpoints"
                    / "last"
                    / "pretrained_model"
                )
                completed = destination.parents[2] / "COMPLETED"
                if (
                    skip_completed
                    and completed.is_file()
                    and (destination / "task_vector_manifest.json").is_file()
                ):
                    print(f"skip completed: {destination}")
                    continue
                build_task_vector_checkpoint(
                    seen_anchor=seen_anchor,
                    target_anchor=target_anchor,
                    seen_partner=seen_partner,
                    target_partner=target_partner,
                    output=destination,
                    partner_weight=alpha,
                )
                completed.write_text("ok\n", encoding="utf-8")
                print(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interpolate only target task vectors across seeds")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[5, 10, 25])
    parser.add_argument("--tasks", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--seeds", type=int, nargs=2, default=[17, 29])
    parser.add_argument("--partner-weight", type=float, default=0.25)
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()
    build_matrix(
        args.output_root,
        args.budgets,
        args.tasks,
        args.seeds,
        args.partner_weight,
        args.skip_completed,
    )


if __name__ == "__main__":
    main()
