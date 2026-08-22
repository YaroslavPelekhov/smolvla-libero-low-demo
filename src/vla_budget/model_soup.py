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
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def interpolate_checkpoints(
    anchor: Path,
    partner: Path,
    output: Path,
    *,
    partner_weight: float,
) -> Path:
    """Create an auditable linear interpolation while retaining anchor metadata."""
    if not 0.0 < partner_weight < 0.5:
        raise ValueError("partner_weight must be strictly between 0 and 0.5")
    anchor = anchor.resolve()
    partner = partner.resolve()
    for source in (anchor, partner):
        if not (source / "model.safetensors").is_file():
            raise FileNotFoundError(f"Missing model.safetensors: {source}")
    if output.exists():
        manifest = output / "soup_manifest.json"
        if not manifest.is_file():
            raise FileExistsError(f"Output exists without soup manifest: {output}")
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        expected = {
            "anchor": str(anchor),
            "partner": str(partner),
            "partner_weight": partner_weight,
        }
        if any(existing.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Existing soup has a different identity: {output}")
        return output

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        shutil.copytree(anchor, temporary, dirs_exist_ok=True)
        model_a = anchor / "model.safetensors"
        model_b = partner / "model.safetensors"
        tensors = {}
        averaged = 0
        copied = 0
        with safe_open(model_a, framework="pt", device="cpu") as left, safe_open(
            model_b, framework="pt", device="cpu"
        ) as right:
            left_keys = set(left.keys())
            right_keys = set(right.keys())
            if left_keys != right_keys:
                raise ValueError("Checkpoint tensor keys do not match")
            for key in sorted(left_keys):
                a = left.get_tensor(key)
                b = right.get_tensor(key)
                if a.shape != b.shape or a.dtype != b.dtype:
                    raise ValueError(f"Tensor mismatch for {key}")
                if torch.is_floating_point(a):
                    tensors[key] = torch.lerp(a.float(), b.float(), partner_weight).to(a.dtype)
                    averaged += a.numel()
                else:
                    if not torch.equal(a, b):
                        raise ValueError(f"Non-floating tensor differs: {key}")
                    tensors[key] = a
                    copied += a.numel()
        save_file(tensors, temporary / "model.safetensors", metadata={"format": "pt"})
        manifest = {
            "schema_version": 1,
            "method": "cross_seed_weight_interpolation",
            "anchor": str(anchor),
            "partner": str(partner),
            "anchor_weight": 1.0 - partner_weight,
            "partner_weight": partner_weight,
            "anchor_sha256": _sha256(model_a),
            "partner_sha256": _sha256(model_b),
            "output_sha256": _sha256(temporary / "model.safetensors"),
            "averaged_parameters": averaged,
            "copied_parameters": copied,
        }
        (temporary / "soup_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def _checkpoint(root: Path, budget: int, task: int, seed: int) -> Path:
    return (
        root
        / "train"
        / "baseline"
        / f"budget-{budget}"
        / f"task-{task}"
        / f"seed-{seed}"
        / "checkpoints"
        / "last"
        / "pretrained_model"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cross-seed interpolated SmolVLA policies")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[5, 10, 25])
    parser.add_argument("--tasks", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--seeds", type=int, nargs=2, default=[17, 29])
    parser.add_argument("--partner-weight", type=float, default=0.25)
    args = parser.parse_args()
    seed_a, seed_b = args.seeds
    for budget in args.budgets:
        for task in args.tasks:
            for anchor_seed, partner_seed in ((seed_a, seed_b), (seed_b, seed_a)):
                source_a = _checkpoint(args.output_root, budget, task, anchor_seed)
                source_b = _checkpoint(args.output_root, budget, task, partner_seed)
                destination = (
                    args.output_root
                    / "train"
                    / "stabilized"
                    / f"budget-{budget}"
                    / f"task-{task}"
                    / f"seed-{anchor_seed}"
                    / "checkpoints"
                    / "last"
                    / "pretrained_model"
                )
                interpolate_checkpoints(
                    source_a,
                    source_b,
                    destination,
                    partner_weight=args.partner_weight,
                )
                marker = destination.parents[2] / "COMPLETED"
                marker.write_text("ok\n", encoding="utf-8")
                print(destination, flush=True)


if __name__ == "__main__":
    main()
