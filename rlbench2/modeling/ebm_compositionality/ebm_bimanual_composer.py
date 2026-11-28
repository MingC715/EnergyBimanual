from torch import nn

from ..policy.bimanual_composer import BimanualComposer
from .bimanual_coordination_constraints import BimanualCoordinationConstraints
from .interaction_energy import CoordinationEnergyModule


class EBMBimanualComposer(BimanualComposer):
    def __init__(
        self, *, use_interaction_energy=False, enable_coordination_constraints=True,
        coord_weight_hidden_dim=64, min_ee_distance=0.001, min_joint_distance=0.001,
        interaction_cond_dim=360, interaction_hidden_dim=128, interaction_num_heads=4,
        lambda_smooth=0.1, use_gate=True, interaction_weight=1.0, min_energy=0.0,
        **actor_config,
    ):
        super().__init__(**actor_config)
        self.energy_composer = nn.Module()
        self.energy_composer.left_actor = self.left_actor
        self.energy_composer.right_actor = self.right_actor
        if use_interaction_energy:
            self.energy_composer.coordination_module = CoordinationEnergyModule(
                action_dim=8, state_dim=8, cond_dim=interaction_cond_dim,
                hidden_dim=interaction_hidden_dim, num_heads=interaction_num_heads,
                lambda_smooth=lambda_smooth, min_ee_distance=min_ee_distance,
                min_joint_distance=min_joint_distance, use_gate=use_gate,
                interaction_weight=interaction_weight, min_energy=min_energy,
            )
        elif enable_coordination_constraints:
            self.energy_composer.coord_constraints = BimanualCoordinationConstraints(
                action_dim=8, min_ee_distance=min_ee_distance, min_joint_distance=min_joint_distance,
                weight_hidden_dim=coord_weight_hidden_dim,
            )
        # Released checkpoints also store these tensors under this second prefix.
        self.energy_sampler = nn.ModuleDict({"energy_composer": self.energy_composer})
