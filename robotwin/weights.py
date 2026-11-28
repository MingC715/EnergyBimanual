import hydra
from omegaconf import OmegaConf
import torch


def load_composed(path, backbone, device="cuda:0"):
    """Restore both arms, normalization buffers and coordination parameters."""
    payload = torch.load(str(path), map_location="cpu", mmap=True, weights_only=True)
    if payload["backbone"] != backbone:
        raise ValueError(f"{path} contains {payload['backbone']}, expected {backbone}")
    if backbone == "dp":
        from diffusion_policy.policy.energy_bimanual_composer import EnergyBimanualComposer as Composer
    elif backbone == "dp3":
        from diffusion_policy_3d.policy.energy_bimanual_dp3 import EnergyBimanualDP3 as Composer
    else:
        raise ValueError(f"Unknown diffusion policy {backbone}")
    left = hydra.utils.instantiate(OmegaConf.create(payload["left_policy"]))
    right = hydra.utils.instantiate(OmegaConf.create(payload["right_policy"]))
    config = payload["composer"]
    fields = ["left_action_dim", "right_action_dim", "coord_hidden_dim"]
    if backbone == "dp":
        fields.extend(["min_ee_distance", "min_joint_distance"])
    policy = Composer(left_policy=left, right_policy=right, **{name: config[name] for name in fields})
    policy.load_state_dict(payload["weight"], strict=True)
    return policy.to(device).eval()
