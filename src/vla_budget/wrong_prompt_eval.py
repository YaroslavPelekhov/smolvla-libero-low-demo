from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate with a deliberately wrong policy prompt")
    parser.add_argument("--wrong-prompt", required=True)
    parser.add_argument("remainder", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    remainder = args.remainder[1:] if args.remainder[:1] == ["--"] else args.remainder
    if "--eval.use_async_envs=false" not in remainder:
        raise SystemExit("Wrong-prompt control requires synchronous environments")

    from lerobot.envs.libero import LiberoEnv

    original_init = LiberoEnv.__init__

    def patched_init(self, *positional, **keyword):
        original_init(self, *positional, **keyword)
        self.task_description = args.wrong_prompt

    LiberoEnv.__init__ = patched_init
    sys.argv = [sys.argv[0], *remainder]
    from lerobot.scripts.lerobot_eval import main as lerobot_main

    lerobot_main()


if __name__ == "__main__":
    main()

