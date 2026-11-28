import numpy as np
from diffusion_policy.env_runner.dp_runner import DPRunner
from weights import load_composed


class EnergyBimanualDPModel:

    def __init__(self, ckpt_file: str, n_obs_steps: int, n_action_steps: int):
        self.policy = self._load_policy(ckpt_file)
        self.runner = DPRunner(n_obs_steps=n_obs_steps, n_action_steps=n_action_steps)

    def _load_policy(self, checkpoint_path):
        return load_composed(checkpoint_path, "dp")

    def update_obs(self, observation):
        self.runner.update_obs(observation)

    def reset_obs(self):
        self.runner.reset_obs()

    def get_action(self, observation=None):
        action = self.runner.get_action(self.policy, observation)
        return action


def encode_obs(observation):
    head_cam = np.moveaxis(
        observation["observation"]["head_camera"]["rgb"], -1, 0
    ) / 255.0
    left_cam = np.moveaxis(
        observation["observation"]["left_camera"]["rgb"], -1, 0
    ) / 255.0
    right_cam = np.moveaxis(
        observation["observation"]["right_camera"]["rgb"], -1, 0
    ) / 255.0
    obs = dict(
        head_cam=head_cam,
        left_cam=left_cam,
        right_cam=right_cam,
    )
    obs["agent_pos"] = observation["joint_action"]["vector"]
    return obs


def get_model(usr_args):
    return EnergyBimanualDPModel(usr_args["checkpoint"], n_obs_steps=3, n_action_steps=6)


def eval(TASK_ENV, model, observation):
    obs = encode_obs(observation)

    actions = model.get_action(obs)

    for action in actions:
        TASK_ENV.take_action(action)
        if TASK_ENV.eval_success:
            return
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)


def reset_model(model):
    model.reset_obs()
