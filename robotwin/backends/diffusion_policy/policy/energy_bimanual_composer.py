import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


from diffusion_policy.policy.panda_fk import DualPandaFK


class CoordinationConstraints(nn.Module):

    def __init__(
        self,
        action_dim: int = 8,
        hidden_dim: int = 64,
        min_ee_distance: float = 0.001,
        min_joint_distance: float = 0.001,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.joint_dim = action_dim - 1
        self.min_ee_distance = min_ee_distance
        self.min_joint_distance = min_joint_distance

        self.weight_net = nn.Sequential(
            nn.Linear(action_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 5),
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.constraint_gate = nn.Parameter(torch.ones(5))

        self.dual_fk = DualPandaFK()

    def forward(
        self, left_actions: torch.Tensor, right_actions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        B, T, D = left_actions.shape

        E_jerk = self._compute_jerk(left_actions, right_actions)
        E_vel = self._compute_velocity_smoothness(left_actions, right_actions)
        E_sync = self._compute_temporal_sync(left_actions, right_actions)
        E_ee_col = self._compute_ee_collision(left_actions, right_actions)
        E_joint_col = self._compute_joint_collision(left_actions, right_actions)

        combined = torch.cat([left_actions, right_actions], dim=-1)
        weights = F.softmax(
            self.weight_net(combined) / self.temperature.clamp(min=0.1), dim=-1
        )
        gate = torch.sigmoid(self.constraint_gate)

        total = (
            gate[0] * weights[..., 0] * E_jerk
            + gate[1] * weights[..., 1] * E_vel
            + gate[2] * weights[..., 2] * E_sync
            + gate[3] * weights[..., 3] * E_ee_col
            + gate[4] * weights[..., 4] * E_joint_col
        )

        return {
            "total": total.mean(dim=1),
            "jerk": E_jerk.mean(dim=1),
            "velocity_smoothness": E_vel.mean(dim=1),
            "temporal_sync": E_sync.mean(dim=1),
            "ee_collision": E_ee_col.mean(dim=1),
            "joint_collision": E_joint_col.mean(dim=1),
        }

    def _compute_jerk(self, left, right):
        B, T, D = left.shape
        if T < 4:
            return torch.zeros(B, T, device=left.device)
        lp = left[..., :self.joint_dim]
        rp = right[..., :self.joint_dim]
        lv = lp[:, 1:] - lp[:, :-1]
        rv = rp[:, 1:] - rp[:, :-1]
        la = lv[:, 1:] - lv[:, :-1]
        ra = rv[:, 1:] - rv[:, :-1]
        lj = la[:, 1:] - la[:, :-1]
        rj = ra[:, 1:] - ra[:, :-1]
        energy = lj.pow(2).sum(-1) + rj.pow(2).sum(-1)
        pad_size = T - energy.shape[1]
        return F.pad(energy, (pad_size, 0), value=0.0)

    def _compute_velocity_smoothness(self, left, right):
        B, T, D = left.shape
        if T < 3:
            return torch.zeros(B, T, device=left.device)
        lp = left[..., :self.joint_dim]
        rp = right[..., :self.joint_dim]
        lv = lp[:, 1:] - lp[:, :-1]
        rv = rp[:, 1:] - rp[:, :-1]
        la = lv[:, 1:] - lv[:, :-1]
        ra = rv[:, 1:] - rv[:, :-1]
        energy = la.pow(2).sum(-1) + ra.pow(2).sum(-1)
        pad_size = T - energy.shape[1]
        return F.pad(energy, (pad_size, 0), value=0.0)

    def _compute_temporal_sync(self, left, right):
        B, T, D = left.shape
        if T < 2:
            return torch.zeros(B, T, device=left.device)
        lp = left[..., :self.joint_dim]
        rp = right[..., :self.joint_dim]
        lv = lp[:, 1:] - lp[:, :-1]
        rv = rp[:, 1:] - rp[:, :-1]
        ls = lv.norm(dim=-1)
        rs = rv.norm(dim=-1)
        speed_diff = (ls - rs).abs()
        lv_norm = F.normalize(lv + 1e-8, dim=-1)
        rv_norm = F.normalize(rv + 1e-8, dim=-1)
        cosine = (lv_norm * rv_norm).sum(dim=-1)
        direction_diff = 1.0 - cosine.abs()
        sync_energy = speed_diff + direction_diff
        pad_size = T - sync_energy.shape[1]
        return F.pad(sync_energy, (pad_size, 0), value=0.0)

    def _compute_ee_collision(self, left, right):
        B, T, D = left.shape
        left_joints = left[..., :self.joint_dim].reshape(B * T, self.joint_dim)
        right_joints = right[..., :self.joint_dim].reshape(B * T, self.joint_dim)
        left_ee, right_ee = self.dual_fk.compute_ee_positions(left_joints, right_joints)
        left_ee = left_ee.reshape(B, T, 3)
        right_ee = right_ee.reshape(B, T, 3)
        distance = (left_ee - right_ee).norm(dim=-1)
        collision_energy = torch.exp(-distance / self.min_ee_distance)
        violation = F.relu(self.min_ee_distance - distance)
        return collision_energy + violation * 20.0

    def _compute_joint_collision(self, left, right):
        B, T, D = left.shape
        left_joints = left[..., :self.joint_dim].reshape(B * T, self.joint_dim)
        right_joints = right[..., :self.joint_dim].reshape(B * T, self.joint_dim)
        left_all, right_all = self.dual_fk.compute_all_joint_positions(
            left_joints, right_joints
        )
        left_all = left_all.reshape(B, T, 8, 3)
        right_all = right_all.reshape(B, T, 8, 3)
        elbow_idx = 4
        wrist_idx = 6
        elbow_dist = (left_all[:, :, elbow_idx] - right_all[:, :, elbow_idx]).norm(dim=-1)
        wrist_dist = (left_all[:, :, wrist_idx] - right_all[:, :, wrist_idx]).norm(dim=-1)
        elbow_energy = torch.exp(-elbow_dist / self.min_joint_distance)
        wrist_energy = torch.exp(-wrist_dist / self.min_joint_distance)
        elbow_violation = F.relu(self.min_joint_distance - elbow_dist) * 20.0
        wrist_violation = F.relu(self.min_joint_distance - wrist_dist) * 20.0
        return elbow_energy + wrist_energy + elbow_violation + wrist_violation


class EnergyBimanualComposer(nn.Module):

    def __init__(self, left_policy, right_policy, left_action_dim=8, right_action_dim=8,
                 coord_hidden_dim=64, min_ee_distance=0.001, min_joint_distance=0.001):
        super().__init__()
        self.left_action_dim = left_action_dim
        self.right_action_dim = right_action_dim
        self.left_policy = left_policy
        self.right_policy = right_policy
        self.coord_constraints = CoordinationConstraints(
            action_dim=max(left_action_dim, right_action_dim), hidden_dim=coord_hidden_dim,
            min_ee_distance=min_ee_distance, min_joint_distance=min_joint_distance,
        )

    def predict_action(self, obs_dict):
        agent_pos = obs_dict["agent_pos"]
        left_agent_pos = agent_pos[..., :self.left_action_dim]
        right_agent_pos = agent_pos[..., self.left_action_dim:]

        left_obs = {k: v for k, v in obs_dict.items() if k != "agent_pos"}
        left_obs["agent_pos"] = left_agent_pos
        right_obs = {k: v for k, v in obs_dict.items() if k != "agent_pos"}
        right_obs["agent_pos"] = right_agent_pos

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
