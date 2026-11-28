import numpy as np
from collections import deque
import torch
from weights import load_composed


class EnergyBimanualDP3Model:

    def __init__(self, ckpt_file: str, n_obs_steps: int, n_action_steps: int):
        self.policy = self._load_policy(ckpt_file)
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.obs = deque(maxlen=n_obs_steps + 1)

    def _load_policy(self, checkpoint_path):
        return load_composed(checkpoint_path, "dp3")

    def stack_last_n_obs(self, all_obs, n_steps):
        all_obs = list(all_obs)
        if isinstance(all_obs[0], np.ndarray):
            result = np.zeros(
                (n_steps,) + all_obs[-1].shape, dtype=all_obs[-1].dtype
            )
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = np.array(all_obs[start_idx:])
            if n_steps > len(all_obs):
                result[:start_idx] = result[start_idx]
        elif isinstance(all_obs[0], torch.Tensor):
            result = torch.zeros(
                (n_steps,) + all_obs[-1].shape, dtype=all_obs[-1].dtype
            )
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = torch.stack(all_obs[start_idx:])
            if n_steps > len(all_obs):
                result[:start_idx] = result[start_idx]
        else:
            raise RuntimeError(f"Unsupported obs type {type(all_obs[0])}")
        return result

    def update_obs(self, observation):
        self.obs.append(observation)

    def reset_obs(self):
        self.obs.clear()

    def get_action(self, observation=None):
        device = self.policy.device
        dtype = self.policy.dtype

        if observation is not None:
            self.obs.append(observation)

        assert len(self.obs) > 0, "No observation recorded"

        result = dict()
        for key in self.obs[0].keys():
            result[key] = self.stack_last_n_obs(
                [obs[key] for obs in self.obs], self.n_obs_steps
            )

        obs_dict = {
            k: torch.from_numpy(v).to(device=device) if isinstance(v, np.ndarray) else v.to(device=device)
            for k, v in result.items()
        }

        with torch.no_grad():
            obs_dict_input = {
                "point_cloud": obs_dict["point_cloud"].unsqueeze(0),
                "agent_pos": obs_dict["agent_pos"].unsqueeze(0),
            }
            action_dict = self.policy.predict_action(obs_dict_input)

        action = action_dict["action"].detach().cpu().numpy().squeeze(0)
        return action


def encode_obs(observation):
    obs = dict()
    obs["agent_pos"] = observation["joint_action"]["vector"]
    obs["point_cloud"] = observation["pointcloud"]
    return obs


def get_model(usr_args):
    return EnergyBimanualDP3Model(usr_args["checkpoint"], n_obs_steps=3, n_action_steps=6)


def eval(TASK_ENV, model, observation):
    obs = encode_obs(observation)

    if len(model.obs) == 0:
        model.update_obs(obs)

    actions = model.get_action()

    for action in actions:
        TASK_ENV.take_action(action)
        if TASK_ENV.eval_success:
            return
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)


def reset_model(model):
    model.reset_obs()
