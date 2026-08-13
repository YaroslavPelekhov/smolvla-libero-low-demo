from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download

from .config import load_config, resolve


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the two leakage-safe LIBERO suites")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--suite", choices=["seen", "target", "both"], default="both")
    parser.add_argument(
        "--local-dir",
        help="Override dataset.local_dir (use a native Linux path for faster WSL training)",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    data = cfg["dataset"]
    suites = {
        "seen": [data["seen_suite"]],
        "target": [data["target_suite"]],
        "both": [data["seen_suite"], data["target_suite"]],
    }[args.suite]
    patterns = [f"{suite}/meta/**" for suite in suites]
    if not args.metadata_only:
        patterns = [f"{suite}/**" for suite in suites]
    destination = resolve(args.local_dir or data["local_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=data["repo_id"],
        repo_type="dataset",
        revision=data.get("revision", "main"),
        local_dir=destination,
        allow_patterns=patterns,
    )
    print(destination)


if __name__ == "__main__":
    main()
