from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordinationConstraints(nn.Module):
    """Score smoothness, synchrony and joint separation."""

    def __init__(self, action_dim: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.action_dim = action_dim
        self.joint_dim = action_dim - 1
        self.weight_net = nn.Sequential(
            nn.Linear(action_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.constraint_gate = nn.Parameter(torch.zeros(3))

    def forward(
        self, left_actions: torch.Tensor, right_actions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        smoothness = self._compute_smoothness(left_actions, right_actions)
        synchrony = self._compute_synchrony(left_actions, right_actions)
        separation = self._compute_joint_separation(left_actions, right_actions)

        combined = torch.cat([left_actions, right_actions], dim=-1)
        weights = F.softmax(
            self.weight_net(combined) / self.temperature.clamp(min=0.1), dim=-1
        )
        gates = torch.sigmoid(self.constraint_gate).view(1, 1, 3)
        weights = weights * gates

        terms = torch.stack([smoothness, synchrony, separation], dim=-1)
        weighted_terms = weights * terms
        total = weighted_terms.sum(dim=-1)

        return {
            "total": total.mean(dim=1),
            "smoothness": smoothness.mean(dim=1),
            "synchrony": synchrony.mean(dim=1),
            "separation": separation.mean(dim=1),
            "weight_smoothness": weights[..., 0].mean(dim=1),
            "weight_synchrony": weights[..., 1].mean(dim=1),
            "weight_separation": weights[..., 2].mean(dim=1),
        }

    def _compute_smoothness(
        self, left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        batch, horizon, _ = left.shape
        if horizon < 3:
            return torch.zeros(batch, horizon, device=left.device, dtype=left.dtype)

        left_joints = left[..., : self.joint_dim]
        right_joints = right[..., : self.joint_dim]
        left_vel = left_joints[:, 1:] - left_joints[:, :-1]
        right_vel = right_joints[:, 1:] - right_joints[:, :-1]
        left_acc = left_vel[:, 1:] - left_vel[:, :-1]
        right_acc = right_vel[:, 1:] - right_vel[:, :-1]
        energy = left_acc.pow(2).sum(-1) + right_acc.pow(2).sum(-1)
        return F.pad(energy, (horizon - energy.shape[1], 0), value=0.0)

    def _compute_synchrony(
        self, left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        batch, horizon, _ = left.shape
        if horizon < 2:
            return torch.zeros(batch, horizon, device=left.device, dtype=left.dtype)

        left_joints = left[..., : self.joint_dim]
        right_joints = right[..., : self.joint_dim]
        left_vel = left_joints[:, 1:] - left_joints[:, :-1]
        right_vel = right_joints[:, 1:] - right_joints[:, :-1]
        speed_diff = (left_vel.norm(dim=-1) - right_vel.norm(dim=-1)).abs()
        direction_diff = 1.0 - (
            F.normalize(left_vel + 1e-8, dim=-1)
            * F.normalize(right_vel + 1e-8, dim=-1)
        ).sum(dim=-1).abs()
        energy = speed_diff + direction_diff
        return F.pad(energy, (horizon - energy.shape[1], 0), value=0.0)

    def _compute_joint_separation(
        self, left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        left_joints = left[..., : self.joint_dim]
        right_joints = right[..., : self.joint_dim]
        distance = (left_joints - right_joints).norm(dim=-1)
        return torch.exp(-distance)


class EnergyBimanualDP3(nn.Module):

    def __init__(self, left_policy, right_policy, left_action_dim=8, right_action_dim=8,
                 coord_hidden_dim=64):
        super().__init__()
        self.left_action_dim = left_action_dim
        self.right_action_dim = right_action_dim
        self.left_policy = left_policy
        self.right_policy = right_policy
        self.coord_constraints = CoordinationConstraints(
            action_dim=max(left_action_dim, right_action_dim), hidden_dim=coord_hidden_dim,
        )

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        agent_pos = obs_dict["agent_pos"]
        left_obs = {k: v for k, v in obs_dict.items() if k != "agent_pos"}
        left_obs["agent_pos"] = agent_pos[..., : self.left_action_dim]
        right_obs = {k: v for k, v in obs_dict.items() if k != "agent_pos"}
        right_obs["agent_pos"] = agent_pos[..., self.left_action_dim :]

        left_result = self.left_policy.predict_action(left_obs)
        right_result = self.right_policy.predict_action(right_obs)

        action = torch.cat([left_result["action"], right_result["action"]], dim=-1)
        action_pred = torch.cat(
            [left_result["action_pred"], right_result["action_pred"]], dim=-1
        )
        return {"action": action, "action_pred": action_pred}

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
