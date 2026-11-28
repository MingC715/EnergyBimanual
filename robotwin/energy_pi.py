from pathlib import Path

import jax.numpy as jnp
import numpy as np

from openpi import transforms
from openpi.models import model as base_model
from openpi.models.energy_composer import EnergyComposedPi0Config
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.policies.aloha_policy import AlohaInputs, AlohaOutputs
from openpi.policies.policy import Policy
from openpi.shared import normalize


class ComposedPi:
    def __init__(self, checkpoint, family, demos):
        checkpoint = Path(checkpoint)
        pi05 = family == "pi05"
        config = EnergyComposedPi0Config(
            pi05=pi05, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
            composed_action_dim=14, single_arm_action_dim=7,
        )
        model = config.load(base_model.restore_params(checkpoint / "params", dtype=jnp.bfloat16))
        stats = normalize.load(checkpoint / "assets" / f"aloha_dual_arm_{demos}")
        mask = transforms.make_bool_mask(6, -1, 6, -1)
        self.policy = Policy(
            model,
            transforms=[
                transforms.InjectDefaultPrompt(None), AlohaInputs(adapt_to_pi=False, action_dim=14),
                transforms.DeltaActions(mask), transforms.Normalize(stats, use_quantiles=pi05),
                transforms.InjectDefaultPrompt(None), transforms.ResizeImages(224, 224),
                transforms.TokenizePrompt(PaligemmaTokenizer(config.max_token_len), discrete_state_input=pi05),
                transforms.PadStatesAndActions(config.action_dim),
            ],
            output_transforms=[
                transforms.Unnormalize(stats, use_quantiles=pi05),
                transforms.AbsoluteActions(mask), AlohaOutputs(adapt_to_pi=False, action_dim=14),
            ],
        )
        self.observation_window = None
        self.instruction = None
        self.pi0_step = 50
    def set_language(self, instruction):
        """Set language instruction for the task"""
        self.instruction = instruction

    def update_observation_window(self, img_arr, state):
        """
        Update the observation window buffer
        
        Args:
            img_arr: List of [front_img, right_img, left_img]
            state: Robot state array
        """
        img_front, img_right, img_left = img_arr[0], img_arr[1], img_arr[2]

        # Transpose images from HWC to CHW format
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        self.observation_window = {
            "state": state,
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.instruction,
        }

    def get_action(self):
        """Get action from the policy"""
        assert self.observation_window is not None, "update observation_window first!"
        return self.policy.infer(self.observation_window)["actions"]

    def reset_obsrvationwindows(self):
        """Reset observation window and instruction"""
        self.instruction = None
        self.observation_window = None


def encode_obs(observation):
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]

    return input_rgb_arr, input_state


def get_model(usr_args):
    return ComposedPi(usr_args["checkpoint"], usr_args["family"], usr_args["demos"])


def eval(TASK_ENV, model, observation):

    if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)

    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state)


    actions = model.get_action()[:model.pi0_step]

    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)


def reset_model(model):
    model.reset_obsrvationwindows()
