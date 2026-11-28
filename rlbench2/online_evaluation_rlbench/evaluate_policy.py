"""Evaluate a bimanual RLBench policy."""

import argparse
import random
from pathlib import Path
import json
import os

import torch
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modeling.ebm_compositionality import EBMBimanualComposer
from utils.common_utils import str2bool, str_none, round_floats

def eval_log_verbose(args):
    return os.environ.get("EVAL_VERBOSE", "").strip().lower() in {"1", "true", "yes", "on"}


def parse_arguments():
    parser = argparse.ArgumentParser("Bimanual RLBench evaluation")
    # Tuples: (name, type, default)
    arguments = [
        # Testing arguments
        ('checkpoint', str_none, None),
        ('task', str, "close_jar"),
        ('max_tries', int, 10),
        ('max_steps', int, 25),
        ('headless', str2bool, False),
        ('collision_checking', str2bool, False),
        ('seed', int, 0),
        # Dataset arguments
        ('data_dir', Path, Path(__file__).parent / "demos"),
        ('dataset', str, "Peract"),
        ('image_size', str, "256,256"),
        # Logging arguments
        ('output_file', Path, Path(__file__).parent / "eval.json"),
        # Model arguments: general policy type
        ('model_type', str, 'ebm_bimanual_composer'),
        ('bimanual', str2bool, True),
        ('prediction_len', int, 1),
        # Model arguments: encoder
        ('backbone', str, "clip"),
        ('fps_subsampling_factor', int, 5),
        # Model arguments: encoder and head
        ('embedding_dim', int, 120),
        ('num_attn_heads', int, 8),
        ('num_vis_instr_attn_layers', int, 2),
        ('num_history', int, 3),
        # Model arguments: head
        ('num_shared_attn_layers', int, 4),
        ('relative_action', str2bool, False),
        ('rotation_format', str, 'quat_xyzw'),
        ('denoise_timesteps', int, 5),
        ('denoise_model', str, "rectified_flow"),
        # Coordination constraint parameters
        ('enable_coordination_constraints', str2bool, False),
        ('coord_weight_hidden_dim', int, 64),
        ('min_ee_distance', float, 0.001),
        ('min_joint_distance', float, 0.001),
        ('use_interaction_energy', str2bool, False),
        ('lambda_smooth', float, 0.1),
        ('interaction_hidden_dim', int, 128),
        ('interaction_num_heads', int, 4),
        ('interaction_cond_dim', int, 360),
        ('use_gate', str2bool, True),
        ('interaction_weight', float, 1.0),
        ('min_energy', float, 0.0),
    ]
    for arg in arguments:
        parser.add_argument(f'--{arg[0]}', type=arg[1], default=arg[2])

    return parser.parse_args()


def load_models(args):
    """Load the policy and its checkpoint weights."""
    model = EBMBimanualComposer(
        backbone=args.backbone, num_vis_instr_attn_layers=args.num_vis_instr_attn_layers,
        fps_subsampling_factor=args.fps_subsampling_factor, embedding_dim=args.embedding_dim,
        num_attn_heads=args.num_attn_heads, nhist=args.num_history, nhand=1,
        num_shared_attn_layers=args.num_shared_attn_layers, relative=args.relative_action,
        rotation_format=args.rotation_format, denoise_timesteps=args.denoise_timesteps,
        denoise_model=args.denoise_model,
        enable_coordination_constraints=args.enable_coordination_constraints,
        coord_weight_hidden_dim=args.coord_weight_hidden_dim,
        min_ee_distance=args.min_ee_distance, min_joint_distance=args.min_joint_distance,
        use_interaction_energy=args.use_interaction_energy, lambda_smooth=args.lambda_smooth,
        interaction_hidden_dim=args.interaction_hidden_dim, interaction_num_heads=args.interaction_num_heads,
        interaction_cond_dim=args.interaction_cond_dim, use_gate=args.use_gate,
        interaction_weight=args.interaction_weight, min_energy=args.min_energy,
    )
    checkpoint = torch.load(str(args.checkpoint), map_location="cpu", weights_only=True, mmap=True)
    state = {name.removeprefix("module."): tensor for name, tensor in checkpoint["weight"].items()}
    model.load_state_dict(state, strict=True)
    return model.eval().cuda()


if __name__ == "__main__":
    args = parse_arguments()
    if eval_log_verbose(args):
        print(f"Task {args.task}, seed {args.seed}, max steps {args.max_steps}")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    from online_evaluation_rlbench.utils_with_bimanual_rlbench import RLBenchEnv, Actioner

    model = load_models(args)

    task_success_rates = {}
    for task_str in [args.task]:

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        env = RLBenchEnv(
            data_path=args.data_dir,
            task_str=task_str,
            image_size=[int(x) for x in args.image_size.split(",")],
            apply_rgb=True,
            apply_pc=True,
            headless=bool(args.headless),
            apply_cameras=("front", "wrist_left", "wrist_right"),
            collision_checking=bool(args.collision_checking)
        )

        actioner = Actioner(model, backbone=args.backbone)

        var_success_rates = env.evaluate_task_on_multiple_variations(
            task_str,
            max_steps=args.max_steps,
            actioner=actioner,
            max_tries=args.max_tries,
            prediction_len=args.prediction_len,
            num_history=args.num_history
        )
        print(
            f"{task_str} variation success rates:",
            round_floats(var_success_rates)
        )
        print(
            f"{task_str} mean success rate:",
            round_floats(var_success_rates["mean"])
        )

        task_success_rates[task_str] = var_success_rates
        with open(args.output_file, "w") as f:
            json.dump(round_floats(task_success_rates), f, indent=4)
