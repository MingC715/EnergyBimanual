import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from .panda_kinematics import PandaIK


class InteractionEnergy(nn.Module):
    """Compute a scalar energy by cross-attending the two arms."""

    def __init__(self, action_dim: int = 8, cond_dim: int = 360,
                 hidden_dim: int = 128, num_heads: int = 4,
                 min_energy: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.min_energy = min_energy

        self.encoder_L = nn.Sequential(
            nn.Linear(action_dim + cond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.encoder_R = nn.Sequential(
            nn.Linear(action_dim + cond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.norm_L = nn.LayerNorm(hidden_dim)
        self.norm_R = nn.LayerNorm(hidden_dim)

        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, a_L: torch.Tensor, a_R: torch.Tensor,
                c_L: torch.Tensor, c_R: torch.Tensor) -> torch.Tensor:
        h_L = self.encoder_L(torch.cat([a_L, c_L], dim=-1))
        h_R = self.encoder_R(torch.cat([a_R, c_R], dim=-1))

        h_L_seq = h_L.unsqueeze(1)
        h_R_seq = h_R.unsqueeze(1)

        h_L_cross, _ = self.cross_attn(h_L_seq, h_R_seq, h_R_seq)
        h_L_cross = self.norm_L(h_L_seq + h_L_cross).squeeze(1)

        h_R_cross, _ = self.cross_attn(h_R_seq, h_L_seq, h_L_seq)
        h_R_cross = self.norm_R(h_R_seq + h_R_cross).squeeze(1)

        raw_energy = self.forward_raw(a_L, a_R, c_L, c_R)
        energy = torch.nn.functional.softplus(raw_energy) + self.min_energy
        return energy

    def forward_raw(self, a_L: torch.Tensor, a_R: torch.Tensor,
                    c_L: torch.Tensor, c_R: torch.Tensor) -> torch.Tensor:
        h_L = self.encoder_L(torch.cat([a_L, c_L], dim=-1))
        h_R = self.encoder_R(torch.cat([a_R, c_R], dim=-1))
        h_L_seq = h_L.unsqueeze(1)
        h_R_seq = h_R.unsqueeze(1)
        h_L_cross, _ = self.cross_attn(h_L_seq, h_R_seq, h_R_seq)
        h_L_cross = self.norm_L(h_L_seq + h_L_cross).squeeze(1)
        h_R_cross, _ = self.cross_attn(h_R_seq, h_L_seq, h_L_seq)
        h_R_cross = self.norm_R(h_R_seq + h_R_cross).squeeze(1)
        x = torch.cat([h_L_cross, h_R_cross], dim=-1)
        for layer in self.energy_head:
            if isinstance(layer, nn.Softplus):
                break
            x = layer(x)
        return x.squeeze(-1)


class InteractionGate(nn.Module):

    def __init__(self, state_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, s_L: torch.Tensor, s_R: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s_L, s_R], dim=-1)).squeeze(-1)


class CollisionEnergy(nn.Module):

    def __init__(self, min_ee_distance: float = 0.001,
                 min_joint_distance: float = 0.001):
        super().__init__()
        self.min_ee_distance = min_ee_distance
        self.min_joint_distance = min_joint_distance

        self.ik_left = PandaIK(base_position=[0.0, 0.37, 0.61])
        self.ik_right = PandaIK(base_position=[0.0, -0.37, 0.61])

    def forward(self, a_L: torch.Tensor, a_R: torch.Tensor) -> torch.Tensor:
        squeeze_time = False
        if a_L.dim() == 2:
            a_L = a_L.unsqueeze(1)
            a_R = a_R.unsqueeze(1)
            squeeze_time = True

        E_ee = self._compute_ee_collision(a_L, a_R)
        E_joint = self._compute_joint_collision(a_L, a_R)

        return (E_ee + E_joint).mean(dim=1)


    def _compute_ee_collision(self, left_actions: torch.Tensor,
                              right_actions: torch.Tensor) -> torch.Tensor:
        left_pos = left_actions[..., :3]
        right_pos = right_actions[..., :3]
        distance = (left_pos - right_pos).norm(dim=-1)
        collision_energy = torch.exp(-distance / self.min_ee_distance)
        violation = F.relu(self.min_ee_distance - distance)
        return collision_energy + violation * 20.0

    def _compute_joint_collision(self, left_actions: torch.Tensor,
                                 right_actions: torch.Tensor) -> torch.Tensor:
        B, T = left_actions.shape[:2]

        left_pos = left_actions[..., :3]
        left_quat = left_actions[..., 3:7]
        right_pos = right_actions[..., :3]
        right_quat = right_actions[..., 3:7]

        left_joint_pos = self.ik_left.compute_joint_positions(
            left_pos.reshape(B * T, 3), left_quat.reshape(B * T, 4)
        )
        right_joint_pos = self.ik_right.compute_joint_positions(
            right_pos.reshape(B * T, 3), right_quat.reshape(B * T, 4)
        )

        num_joints = left_joint_pos.shape[1]
        left_joint_pos = left_joint_pos.reshape(B, T, num_joints, 3)
        right_joint_pos = right_joint_pos.reshape(B, T, num_joints, 3)

        left_exp = left_joint_pos.unsqueeze(3)
        right_exp = right_joint_pos.unsqueeze(2)
        pairwise_dist = (left_exp - right_exp).norm(dim=-1)

        collision_energy = torch.exp(-pairwise_dist / self.min_joint_distance)
        max_collision = collision_energy.max(dim=-1)[0].max(dim=-1)[0]

        # Critical pairs: elbow(3) and wrist(6)
        critical_energy = torch.zeros_like(max_collision)
        elbow_idx, wrist_idx = 3, 6
        if num_joints > max(elbow_idx, wrist_idx):
            elbow_dist = (left_joint_pos[:, :, elbow_idx] -
                          right_joint_pos[:, :, elbow_idx]).norm(dim=-1)
            wrist_dist = (left_joint_pos[:, :, wrist_idx] -
                          right_joint_pos[:, :, wrist_idx]).norm(dim=-1)
            critical_energy = 20.0 * (
                F.relu(self.min_joint_distance - elbow_dist) +
                F.relu(self.min_joint_distance - wrist_dist)
            )

        workspace_penalty = (
            F.relu(-left_actions[..., 1]) +
            F.relu(right_actions[..., 1])
        )

        return max_collision + critical_energy + workspace_penalty


def compute_bimanual_smooth_energy(
    action_chunk_L: torch.Tensor,
    action_chunk_R: torch.Tensor,
) -> torch.Tensor:
    B, T, _ = action_chunk_L.shape

    if T <= 1:
        return torch.zeros(B, device=action_chunk_L.device,
                           dtype=action_chunk_L.dtype)

    smooth = torch.zeros(B, device=action_chunk_L.device,
                         dtype=action_chunk_L.dtype)

    for chunk in (action_chunk_L, action_chunk_R):
        # First-order: ||a_{t+1} - a_t||^2
        diff1 = chunk[:, 1:] - chunk[:, :-1]
        smooth = smooth + diff1.pow(2).sum(dim=-1).mean(dim=1)

        if T >= 3:
            # Second-order: ||a_{t+2} - 2*a_{t+1} + a_t||^2
            diff2 = chunk[:, 2:] - 2 * chunk[:, 1:-1] + chunk[:, :-2]
            smooth = smooth + diff2.pow(2).sum(dim=-1).mean(dim=1)

    return smooth


class CoordinationEnergyModule(nn.Module):
    """Combine gated interaction, smoothness and collision energies."""

    def __init__(
        self,
        action_dim: int = 8,
        state_dim: int = 8,
        cond_dim: int = 360,
        hidden_dim: int = 128,
        num_heads: int = 4,
        lambda_smooth: float = 0.1,
        min_ee_distance: float = 0.001,
        min_joint_distance: float = 0.001,
        use_gate: bool = True,
        interaction_weight: float = 1.0,
        min_energy: float = 0.0,
    ):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.use_gate = use_gate
        self.interaction_weight = interaction_weight

        self.interaction_energy = InteractionEnergy(
            action_dim=action_dim, cond_dim=cond_dim,
            hidden_dim=hidden_dim, num_heads=num_heads,
            min_energy=min_energy,
        )
        self.interaction_gate = InteractionGate(state_dim=state_dim) if use_gate else None
        self.collision_energy = CollisionEnergy(
            min_ee_distance=min_ee_distance,
            min_joint_distance=min_joint_distance,
        )

    def forward(
        self,
        a_L: torch.Tensor,
        a_R: torch.Tensor,
        c_L: torch.Tensor,
        c_R: torch.Tensor,
        s_L: torch.Tensor,
        s_R: torch.Tensor,
        action_chunk_L: Optional[torch.Tensor] = None,
        action_chunk_R: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        gate = self.interaction_gate(s_L, s_R) if self.use_gate else torch.ones(s_L.shape[0], device=s_L.device)
        E_interact = self.interaction_energy(a_L, a_R, c_L, c_R)

        if action_chunk_L is not None:
            E_collision = self.collision_energy(action_chunk_L, action_chunk_R)
        else:
            E_collision = self.collision_energy(a_L, a_R)

        if action_chunk_L is not None and action_chunk_L.shape[1] > 1:
            E_smooth = compute_bimanual_smooth_energy(
                action_chunk_L, action_chunk_R
            )
        else:
            E_smooth = torch.zeros_like(E_interact)

        E_coord = (
            self.interaction_weight * gate * E_interact
            + self.lambda_smooth * E_smooth
            + E_collision
        )

        details = {
            "gate": gate,
            "E_interact": E_interact,
            "E_collision": E_collision,
            "E_smooth": E_smooth,
        }
        return E_coord, details
