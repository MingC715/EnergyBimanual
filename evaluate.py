"""Evaluate bimanual policies on RLBench and RoboTwin 2.0."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
RLBENCH_TASKS = (
    "bimanual_push_box", "bimanual_lift_ball", "bimanual_dual_push_buttons",
    "bimanual_pick_plate", "bimanual_put_item_in_drawer", "bimanual_put_bottle_in_fridge",
    "bimanual_handover_item", "bimanual_pick_laptop", "bimanual_straighten_rope",
    "bimanual_sweep_to_dustpan", "bimanual_lift_tray", "bimanual_handover_item_easy",
    "bimanual_take_tray_out_of_oven",
)
ROBOTWIN_TASKS = (
    "place_burger_fries", "grab_roller", "place_bread_skillet",
    "place_can_basket", "place_cans_plasticbox", "put_object_cabinet",
)


def checkpoint_path(root, family, demos):
    if family in ("energyaction", "energybimanual"):
        return root / "RLBench" / f"{family}_{demos}demo.pth"
    if family in ("dp", "dp3"):
        return root / "RoboTwin2.0" / f"{family}_{demos}demo.pt"
    policy = "pi0.5" if family == "pi05" else family
    return root / "RoboTwin2.0" / f"{policy}_{demos}demo"


def rlbench_command(args, task, checkpoint, output):
    newer = args.method == "energybimanual"
    settings = {
        "checkpoint": checkpoint, "task": task, "data_dir": args.data_dir,
        "output_file": output / task / "eval.json", "seed": args.seed, "max_tries": 2,
        "max_steps": 25, "headless": True, "collision_checking": False,
        "model_type": "ebm_bimanual_composer", "bimanual": True, "prediction_len": 1,
        "backbone": "clip", "fps_subsampling_factor": 5, "image_size": "256,256",
        "embedding_dim": 120, "num_attn_heads": 8, "num_vis_instr_attn_layers": 2,
        "num_history": 3, "num_shared_attn_layers": 4, "rotation_format": "quat_xyzw",
        "relative_action": False, "denoise_model": "rectified_flow", "denoise_timesteps": 5,
        "enable_coordination_constraints": True,
        "min_ee_distance": 0.001, "min_joint_distance": 0.001,
        "use_interaction_energy": newer, "use_gate": newer, "lambda_smooth": 0.1,
        "interaction_hidden_dim": 128, "interaction_num_heads": 4, "interaction_cond_dim": 360,
    }
    command = [sys.executable, str(ROOT / "rlbench2/online_evaluation_rlbench/evaluate_policy.py")]
    for name, value in settings.items():
        command.extend(["--" + name, str(value)])
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="benchmark", required=True)
    rl = modes.add_parser("rlbench", help="bimanual RLBench")
    tw = modes.add_parser("robotwin", help="RoboTwin 2.0")
    rl.add_argument("--data-dir", default=os.environ.get("PERACT2_DATA_ROOT"), type=Path,
                    help="evaluation split containing task/variation*/episodes")
    tw.add_argument("--robotwin-root", default=os.environ.get("ROBOTWIN_ROOT"), type=Path,
                    help="RoboTwin checkout containing assets/ and its simulator dependencies")
    tw.add_argument("--episodes", type=int, default=100, help="number of expert-valid evaluation episodes per task")
    tw.add_argument("--video", action="store_true", help="write rollout videos using ffmpeg")
    for p, methods, tasks in [(rl, ("energybimanual", "energyaction"), RLBENCH_TASKS),
                              (tw, ("dp", "dp3", "pi0", "pi05"), ROBOTWIN_TASKS)]:
        p.add_argument("--method", choices=methods, required=True)
        p.add_argument("--demos", type=int, choices=(5, 10, 20), required=True)
        p.add_argument("--task", choices=("all", *tasks), default="all")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--weights", type=Path, default=ROOT / "checkpoints",
                       help="root containing RLBench/ and RoboTwin2.0/")
        p.add_argument("--output", type=Path, default=Path("outputs"))
        p.add_argument("--dry-run", action="store_true", help="show the exact launch commands")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    rlbench = args.benchmark == "rlbench"
    assets = args.data_dir if rlbench else args.robotwin_root
    if assets is None:
        parser.error("set --data-dir or PERACT2_DATA_ROOT" if rlbench else "set --robotwin-root or ROBOTWIN_ROOT")
    assets = assets.expanduser().resolve()
    if rlbench:
        args.data_dir = assets
    elif args.episodes < 1:
        parser.error("--episodes must be positive")
    checkpoint = checkpoint_path(args.weights.expanduser().resolve(), args.method, args.demos)
    if not args.dry_run:
        if not checkpoint.exists():
            parser.error(f"missing checkpoint {checkpoint}")
        if not assets.is_dir() or (not rlbench and not (assets / "assets").is_dir()):
            parser.error(f"missing benchmark files under {assets}")
    output = args.output.expanduser().resolve() / args.benchmark / args.method / f"{args.demos}demo" / f"seed{args.seed}"
    tasks = (RLBENCH_TASKS if rlbench else ROBOTWIN_TASKS) if args.task == "all" else (args.task,)
    for task in tasks:
        if rlbench:
            if not args.dry_run and not any((assets / task).glob("variation*/episodes/episode*")):
                parser.error(f"no evaluation episodes under {assets / task}")
            command = rlbench_command(args, task, checkpoint, output)
        else:
            command = [sys.executable, str(ROOT / "robotwin/run_task.py"), args.method, str(args.demos),
                       task, str(checkpoint), str(assets), str(output), str(args.seed), str(args.episodes)]
            if args.video:
                command.append("--video")
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
