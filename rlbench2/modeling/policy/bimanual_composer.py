import torch
from torch import nn

from .denoise_actor_3d import DenoiseActor


class BimanualComposer(nn.Module):
    def __init__(self, **actor_config):
        super().__init__()
        self.register_buffer("workspace_normalizer", torch.zeros(2, 3))
        self.left_actor = DenoiseActor(**actor_config)
        self.right_actor = DenoiseActor(**actor_config)

    def forward(
        self, gt_trajectory, trajectory_mask, rgb3d, rgb2d, pcd, instruction,
        proprio=None, left_proprioception=None, right_proprioception=None,
        run_inference=True,
    ):
        if left_proprioception is None or right_proprioception is None:
            if proprio is None:
                raise ValueError("Provide a joint proprioception tensor or both arm histories.")
            left_proprioception = proprio[:, :, 0, :]
            right_proprioception = proprio[:, :, 1, :]
        mask = trajectory_mask[..., :1] if trajectory_mask.shape[-1] == 2 else trajectory_mask
        observation = (mask, rgb3d, rgb2d, pcd, instruction)
        left = self.left_actor.compute_trajectory(*observation, left_proprioception)
        right = self.right_actor.compute_trajectory(*observation, right_proprioception)
        return torch.cat((left, right), dim=2)
