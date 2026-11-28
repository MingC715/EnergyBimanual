"""Evaluate one RoboTwin 2.0 task."""
import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=("dp", "dp3", "pi0", "pi05"))
    parser.add_argument("demos", type=int, choices=(5, 10, 20))
    parser.add_argument("task")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("robotwin_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("seed", type=int)
    parser.add_argument("episodes", type=int)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    assets = args.robotwin_root.resolve()
    checkpoint, output = args.checkpoint.resolve(), args.output.resolve()
    embodiment = "aloha" if args.family in ("pi0", "pi05") else "franka"
    suite = root / "suites" / embodiment
    sys.path[:0] = [str(suite), str(root / "backends"), str(root), str(root / "description/utils"), str(assets)]
    os.chdir(assets)
    from evaluate_rollout import main as rollout
    rollout(dict(task_name=args.task, task_config=f"{embodiment}_dual_arm", family=args.family,
                 policy_name="energy_pi" if embodiment == "aloha" else f"energy_{args.family}",
                 instruction_type="unseen", ckpt_setting=f"{args.demos}demo",
                 checkpoint=str(checkpoint), demos=args.demos, seed=args.seed, episodes=args.episodes,
                 video=args.video, output=str(output)))


if __name__ == "__main__":
    main()
