from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from .config import load_config, resolve


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze predictions before experiment execution")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--predictions", default="predictions_v2.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    predictions = resolve(args.predictions)
    raw = predictions.read_bytes()
    payload = {
        "predictions_file": str(Path(args.predictions)),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "recorded_at_utc": datetime.now(UTC).isoformat(),
    }
    output = resolve(cfg["paths"]["preregistration"])
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing["sha256"] != payload["sha256"]:
            raise SystemExit("Refusing to overwrite preregistration with changed predictions")
        print(f"already registered: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
