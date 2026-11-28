import torch
import numpy as np


class RFScheduler:
    """
    Code based on:
    https://github.com/cloneofsimo/minRF/blob/main/advanced/main_t2i.py
    """

    def set_timesteps(self, num_inference_steps, device='cpu'):
        self.timesteps = torch.from_numpy(
            np.arange(num_inference_steps + 1)[1:][::-1].astype(np.float32)
            / num_inference_steps
        ).to(device)
        # Precompute previous timesteps, where prev_t[0] = 0
        self.prev_timesteps = torch.cat((
            self.timesteps[1:],
            torch.zeros(1, device=device, dtype=self.timesteps.dtype)
        ))

    def step(self, model_output, timestep_ind, sample, dt_scale=1.0):
        zt = sample
        vc = model_output

        timestep = self.timesteps[timestep_ind].to(vc.device)
        prev_t = self.prev_timesteps[timestep_ind].to(vc.device)
        dt = (timestep - prev_t) * dt_scale
        pred_prev_sample = zt - dt * vc  # z_t'

        return DummyClass(prev_sample=pred_prev_sample)


class DummyClass:

    def __init__(self, prev_sample):
        self.prev_sample = prev_sample
