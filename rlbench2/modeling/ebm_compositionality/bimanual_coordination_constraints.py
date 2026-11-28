import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from .panda_kinematics import PandaIK


class ConstraintWeightPredictor(nn.Module):

    def __init__(self, action_dim: int = 8, hidden_dim: int = 64):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(action_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 5)
        )

        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, combined_actions: torch.Tensor) -> torch.Tensor:
        logits = self.network(combined_actions)

        weights = F.softmax(logits / self.temperature.clamp(min=0.1), dim=-1)

        return weights


class BimanualCoordinationConstraints(nn.Module):

    def __init__(self,
                 action_dim: int = 8,
                 min_ee_distance: float = 0.15,
                 min_joint_distance: float = 0.12,
                 weight_hidden_dim: int = 64):
        super().__init__()

        self.action_dim = action_dim
        self.min_ee_distance = min_ee_distance
        self.min_joint_distance = min_joint_distance

        self.weight_predictor = ConstraintWeightPredictor(
            action_dim=action_dim,
            hidden_dim=weight_hidden_dim
        )

        self.base_left = [0.0, 0.37, 0.61]
        self.base_right = [0.0, -0.37, 0.61]

        self._base_left_tensor = None
        self._base_right_tensor = None

        self.ik_left = PandaIK(base_position=self.base_left)
        self.ik_right = PandaIK(base_position=self.base_right)

    def _ensure_base_tensors(self, device: torch.device, dtype: torch.dtype):
        if self._base_left_tensor is None:
            self._base_left_tensor = torch.tensor(
                self.base_left, device=device, dtype=dtype
            )
            self._base_right_tensor = torch.tensor(
                self.base_right, device=device, dtype=dtype
            )

    def forward(self, left_actions: torch.Tensor, right_actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._ensure_base_tensors(left_actions.device, left_actions.dtype)

        E_jerk = self._compute_jerk_energy(left_actions, right_actions)
        E_velocity = self._compute_velocity_smoothness(left_actions, right_actions)
        E_temporal_sync = self._compute_temporal_sync(left_actions, right_actions)
        E_ee_collision = self._compute_ee_collision(left_actions, right_actions)
        E_joint_collision = self._compute_joint_collision(left_actions, right_actions)

        combined_actions = torch.cat([left_actions, right_actions], dim=-1)
        weights = self.weight_predictor(combined_actions)

        weighted_energies = {
            'jerk': weights[..., 0] * E_jerk,
            'velocity_smoothness': weights[..., 1] * E_velocity,
            'temporal_sync': weights[..., 2] * E_temporal_sync,
            'ee_collision': weights[..., 3] * E_ee_collision,
            'joint_collision': weights[..., 4] * E_joint_collision,
        }

        total_energy = sum(weighted_energies.values())

        return {
            **weighted_energies,
            'total': total_energy,
            'weights': weights,
            'raw_energies': {
                'jerk': E_jerk,
                'velocity_smoothness': E_velocity,
                'temporal_sync': E_temporal_sync,
                'ee_collision': E_ee_collision,
                'joint_collision': E_joint_collision,
            }
        }

    def _compute_jerk_energy(self, left_actions: torch.Tensor, right_actions: torch.Tensor) -> torch.Tensor:
        B, T = left_actions.shape[:2]

        if T < 4:
            return torch.zeros(B, T, device=left_actions.device, dtype=left_actions.dtype)

        left_pos = left_actions[..., :3]
        right_pos = right_actions[..., :3]

        left_vel = left_pos[:, 1:] - left_pos[:, :-1]
        right_vel = right_pos[:, 1:] - right_pos[:, :-1]

        left_acc = left_vel[:, 1:] - left_vel[:, :-1]
        right_acc = right_vel[:, 1:] - right_vel[:, :-1]

        left_jerk = left_acc[:, 1:] - left_acc[:, :-1]
        right_jerk = right_acc[:, 1:] - right_acc[:, :-1]

        jerk_norm = (left_jerk.pow(2).sum(dim=-1) +
                     right_jerk.pow(2).sum(dim=-1))

        pad_size = T - jerk_norm.shape[1]
        jerk_energy = F.pad(jerk_norm, (pad_size, 0), value=0.0)

        return jerk_energy

    def _compute_velocity_smoothness(self, left_actions: torch.Tensor, right_actions: torch.Tensor) -> torch.Tensor:
        B, T = left_actions.shape[:2]

        if T < 3:
            return torch.zeros(B, T, device=left_actions.device, dtype=left_actions.dtype)

        left_pos = left_actions[..., :3]
        right_pos = right_actions[..., :3]

        left_vel = left_pos[:, 1:] - left_pos[:, :-1]
        right_vel = right_pos[:, 1:] - right_pos[:, :-1]

        left_vel_change = left_vel[:, 1:] - left_vel[:, :-1]
        right_vel_change = right_vel[:, 1:] - right_vel[:, :-1]

        velocity_change_norm = (left_vel_change.pow(2).sum(dim=-1) +
                               right_vel_change.pow(2).sum(dim=-1))

        pad_size = T - velocity_change_norm.shape[1]
        velocity_energy = F.pad(velocity_change_norm, (pad_size, 0), value=0.0)

        return velocity_energy

    def _compute_temporal_sync(self, left_actions: torch.Tensor, right_actions: torch.Tensor) -> torch.Tensor:
        B, T = left_actions.shape[:2]

        if T < 2:
            return torch.zeros(B, T, device=left_actions.device, dtype=left_actions.dtype)

        left_pos = left_actions[..., :3]
        right_pos = right_actions[..., :3]

        left_vel = left_pos[:, 1:] - left_pos[:, :-1]
        right_vel = right_pos[:, 1:] - right_pos[:, :-1]

        left_speed = left_vel.norm(dim=-1)
        right_speed = right_vel.norm(dim=-1)

        speed_diff = (left_speed - right_speed).abs()

        left_vel_norm = F.normalize(left_vel + 1e-8, dim=-1)
        right_vel_norm = F.normalize(right_vel + 1e-8, dim=-1)

        vel_cosine = (left_vel_norm * right_vel_norm).sum(dim=-1)
        direction_diff = 1.0 - vel_cosine.abs()

        sync_energy = speed_diff + direction_diff

        pad_size = T - sync_energy.shape[1]
        sync_energy = F.pad(sync_energy, (pad_size, 0), value=0.0)

        return sync_energy

    def _compute_ee_collision(self, left_actions: torch.Tensor, right_actions: torch.Tensor) -> torch.Tensor:
        left_pos = left_actions[..., :3]
        right_pos = right_actions[..., :3]

        distance = (left_pos - right_pos).norm(dim=-1)

        collision_energy = torch.exp(-distance / self.min_ee_distance)

        violation = F.relu(self.min_ee_distance - distance)
        collision_energy = collision_energy + violation * 20.0

        return collision_energy

    def _compute_joint_collision(self, left_actions: torch.Tensor, right_actions: torch.Tensor) -> torch.Tensor:
        B, T = left_actions.shape[:2]

        left_pos = left_actions[..., :3]
        left_quat = left_actions[..., 3:7]
        right_pos = right_actions[..., :3]
        right_quat = right_actions[..., 3:7]

        left_pos_flat = left_pos.reshape(B * T, 3)
        left_quat_flat = left_quat.reshape(B * T, 4)
        right_pos_flat = right_pos.reshape(B * T, 3)
        right_quat_flat = right_quat.reshape(B * T, 4)

        left_joint_pos = self.ik_left.compute_joint_positions(left_pos_flat, left_quat_flat)
        right_joint_pos = self.ik_right.compute_joint_positions(right_pos_flat, right_quat_flat)

        num_joints = left_joint_pos.shape[1]
        left_joint_pos = left_joint_pos.reshape(B, T, num_joints, 3)
        right_joint_pos = right_joint_pos.reshape(B, T, num_joints, 3)

        left_expanded = left_joint_pos.unsqueeze(3)
        right_expanded = right_joint_pos.unsqueeze(2)

        pairwise_distances = (left_expanded - right_expanded).norm(dim=-1)

        collision_energy = torch.exp(-pairwise_distances / self.min_joint_distance)

        max_collision_per_timestep = collision_energy.max(dim=-1)[0].max(dim=-1)[0]

        # Focus on elbow (joint 3) and wrist (joint 6) as they're most likely to collide
        elbow_idx = 3
        wrist_idx = 6

        critical_pairs_energy = 0.0
        if num_joints > max(elbow_idx, wrist_idx):
            elbow_distance = (left_joint_pos[:, :, elbow_idx] -
                            right_joint_pos[:, :, elbow_idx]).norm(dim=-1)
            elbow_violation = F.relu(self.min_joint_distance - elbow_distance)

            wrist_distance = (left_joint_pos[:, :, wrist_idx] -
                            right_joint_pos[:, :, wrist_idx]).norm(dim=-1)
            wrist_violation = F.relu(self.min_joint_distance - wrist_distance)

            critical_pairs_energy = 20.0 * (elbow_violation + wrist_violation)

        # Left arm should stay in y > 0 region, right arm in y < 0 region
        workspace_penalty = (
            F.relu(-left_pos[..., 1]) +
            F.relu(right_pos[..., 1])
        )

        joint_collision_energy = (
            max_collision_per_timestep +
            critical_pairs_energy +
            workspace_penalty
        )

        return joint_collision_energy
