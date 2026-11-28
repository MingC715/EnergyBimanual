import torch
import torch.nn as nn
import math
from typing import List, Optional, Tuple


class PandaFK(nn.Module):

    DH_PARAMS = [
        (0.0,     0.333,  0.0),
        (0.0,     0.0,   -math.pi / 2),
        (0.0,     0.316,  math.pi / 2),
        (0.0825,  0.0,    math.pi / 2),
        (-0.0825, 0.384, -math.pi / 2),
        (0.0,     0.0,    math.pi / 2),
        (0.088,   0.107,  math.pi / 2),
    ]

    BASE_LEFT = [0.0, 0.37, 0.61]
    BASE_RIGHT = [0.0, -0.37, 0.61]

    def __init__(self, base_position: Optional[List[float]] = None):
        super().__init__()
        if base_position is None:
            base_position = [0.0, 0.0, 0.0]
        self.register_buffer(
            "base_pos", torch.tensor(base_position, dtype=torch.float32)
        )

        a_list, d_list, alpha_list = zip(*self.DH_PARAMS)
        self.register_buffer("a", torch.tensor(a_list, dtype=torch.float32))
        self.register_buffer("d", torch.tensor(d_list, dtype=torch.float32))
        self.register_buffer(
            "cos_alpha", torch.cos(torch.tensor(alpha_list, dtype=torch.float32))
        )
        self.register_buffer(
            "sin_alpha", torch.sin(torch.tensor(alpha_list, dtype=torch.float32))
        )

    def _dh_transform(
        self, theta: torch.Tensor, joint_idx: int
    ) -> torch.Tensor:
        B = theta.shape[0]
        device = theta.device
        dtype = theta.dtype

        ct = torch.cos(theta)
        st = torch.sin(theta)
        ca = self.cos_alpha[joint_idx].to(dtype=dtype)
        sa = self.sin_alpha[joint_idx].to(dtype=dtype)
        a_i = self.a[joint_idx].to(dtype=dtype)
        d_i = self.d[joint_idx].to(dtype=dtype)

        T = torch.zeros(B, 4, 4, device=device, dtype=dtype)
        T[:, 0, 0] = ct
        T[:, 0, 1] = -st
        T[:, 0, 2] = 0.0
        T[:, 0, 3] = a_i
        T[:, 1, 0] = st * ca
        T[:, 1, 1] = ct * ca
        T[:, 1, 2] = -sa
        T[:, 1, 3] = -sa * d_i
        T[:, 2, 0] = st * sa
        T[:, 2, 1] = ct * sa
        T[:, 2, 2] = ca
        T[:, 2, 3] = ca * d_i
        T[:, 3, 3] = 1.0

        return T

    def compute_joint_positions(
        self, joint_angles: torch.Tensor
    ) -> torch.Tensor:
        B = joint_angles.shape[0]
        device = joint_angles.device
        dtype = joint_angles.dtype

        base = self.base_pos.to(device=device, dtype=dtype)

        T_current = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1).clone()
        T_current[:, 0, 3] = base[0]
        T_current[:, 1, 3] = base[1]
        T_current[:, 2, 3] = base[2]

        positions = [base.unsqueeze(0).expand(B, -1)]

        for i in range(7):
            T_joint = self._dh_transform(joint_angles[:, i], i)
            T_current = torch.bmm(T_current, T_joint)
            pos = T_current[:, :3, 3]  # (B, 3)
            positions.append(pos)

        # (B, 8, 3): base + 7 joints
        return torch.stack(positions, dim=1)

    def compute_ee_position(
        self, joint_angles: torch.Tensor
    ) -> torch.Tensor:
        positions = self.compute_joint_positions(joint_angles)
        return positions[:, -1, :]


class DualPandaFK(nn.Module):

    def __init__(
        self,
        base_left: Optional[List[float]] = None,
        base_right: Optional[List[float]] = None,
    ):
        super().__init__()
        self.left_fk = PandaFK(base_left or PandaFK.BASE_LEFT)
        self.right_fk = PandaFK(base_right or PandaFK.BASE_RIGHT)

    def compute_all_joint_positions(
        self,
        left_joint_angles: torch.Tensor,
        right_joint_angles: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        left_pos = self.left_fk.compute_joint_positions(left_joint_angles)
        right_pos = self.right_fk.compute_joint_positions(right_joint_angles)
        return left_pos, right_pos

    def compute_ee_positions(
        self,
        left_joint_angles: torch.Tensor,
        right_joint_angles: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        left_ee = self.left_fk.compute_ee_position(left_joint_angles)
        right_ee = self.right_fk.compute_ee_position(right_joint_angles)
        return left_ee, right_ee
