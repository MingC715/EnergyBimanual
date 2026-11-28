from typing import List

import torch
from torch import nn


class PandaIK(nn.Module):
    """Approximate seven Panda joint positions from the base and end-effector positions."""

    def __init__(self, base_position: List[float] = None):
        super().__init__()

        if base_position is None:
            base_position = [0.0, 0.0, 0.0]

        self.register_buffer('base_pos', torch.tensor(base_position, dtype=torch.float32))

        self.joint_z_offsets = [
            0.333,
            0.333,
            0.649,
            0.649,
            1.033,
            1.033,
            1.140,
        ]

    def compute_joint_positions(self, ee_pos: torch.Tensor, ee_quat: torch.Tensor) -> torch.Tensor:
        B = ee_pos.shape[0]
        device = ee_pos.device
        dtype = ee_pos.dtype

        base_pos = self.base_pos.to(device=device, dtype=dtype)

        ee_direction = ee_pos - base_pos

        joint_positions = []


        for i, z_offset in enumerate(self.joint_z_offsets):
            if i < 2:
                ratio = 0.1 + (i / 7.0) * 0.2
            elif i < 4:
                ratio = 0.35 + ((i - 2) / 7.0) * 0.15
            else:
                ratio = 0.65 + ((i - 4) / 7.0) * 0.25

            joint_pos = base_pos + ratio * ee_direction

            joint_pos_with_offset = joint_pos.clone()
            joint_pos_with_offset[:, 2] += z_offset * 0.15

            if i == 2 or i == 3:
                perp_xy = torch.zeros_like(ee_direction)
                perp_xy[:, 0] = -ee_direction[:, 1]
                perp_xy[:, 1] = ee_direction[:, 0]
                perp_xy_norm = perp_xy / (perp_xy.norm(dim=-1, keepdim=True) + 1e-8)

                joint_pos_with_offset = joint_pos_with_offset + 0.08 * perp_xy_norm
                joint_pos_with_offset[:, 2] -= 0.05

            joint_positions.append(joint_pos_with_offset)

        joint_positions = torch.stack(joint_positions, dim=1)

        return joint_positions
