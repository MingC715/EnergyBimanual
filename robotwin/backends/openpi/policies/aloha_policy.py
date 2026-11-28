import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms




@dataclasses.dataclass(frozen=True)
class AlohaInputs(transforms.DataTransformFn):
    """Inputs for the Aloha policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [action_dim]
    """

    adapt_to_pi: bool = True

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")

    # The action dimension. For standard Aloha dual-arm, this is 14 (6 joints + 1 gripper per arm).
    # For standard Aloha single-arm, this is 7 (6 joints + 1 gripper).
    # For Franka dual-arm, this is 16 (7 joints + 1 gripper per arm).
    action_dim: int = 14

    def __call__(self, data: dict) -> dict:
        data = _decode_aloha(data, adapt_to_pi=self.adapt_to_pi, action_dim=self.action_dim)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Assume that base image always exists.
        base_image = in_images["cam_high"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # Add the extra images.
        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class AlohaOutputs(transforms.DataTransformFn):
    """Outputs for the Aloha policy."""

    adapt_to_pi: bool = True

    # The action dimension. For standard Aloha dual-arm, this is 14 (6 joints + 1 gripper per arm).
    # For standard Aloha single-arm, this is 7 (6 joints + 1 gripper).
    # For Franka dual-arm, this is 16 (7 joints + 1 gripper per arm).
    action_dim: int = 14

    def __call__(self, data: dict) -> dict:
        # Only return the first action_dim dims.
        actions = np.asarray(data["actions"][:, :self.action_dim])
        return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi, action_dim=self.action_dim)}


def _joint_flip_mask() -> np.ndarray:
    """Used to convert between aloha and pi joint angles."""
    return np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])


def _single_arm_joint_flip_mask() -> np.ndarray:
    """Used to convert single-arm Aloha joint angles between Aloha and pi spaces."""
    return np.array([1, -1, -1, 1, 1, 1, 1])


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with pi0 which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return np.arcsin(np.clip(value, -1.0, 1.0))

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # pi0 gripper data is normalized (0, 1) between encoder counts (2405, 3110).
    # There are 4096 total encoder counts and aloha uses a zero of 2048.
    # Converting this to radians means that the normalized inputs are between (0.5476, 1.6296)
    return _normalize(value, min_val=0.5476, max_val=1.6296)


def _gripper_from_angular(value):
    # Convert from the gripper position used by pi0 to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # Trossen gripper predictions are already in radians.
    value = value + 0.5476

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return _normalize(value, min_val=-0.6213, max_val=1.4910)


def _gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return value - 0.5476


def _decode_aloha(data: dict, *, adapt_to_pi: bool = False, action_dim: int = 14) -> dict:
    # state is [left_arm_joint_angles, left_arm_gripper, right_arm_joint_angles, right_arm_gripper]
    # For standard Aloha single-arm: dim sizes: [6, 1] = 7
    # For standard Aloha dual-arm: dim sizes: [6, 1, 6, 1] = 14
    # For Franka: dim sizes: [7, 1, 7, 1] = 16
    state = np.asarray(data["state"])
    state = _decode_state(state, adapt_to_pi=adapt_to_pi, action_dim=action_dim)

    def convert_image(img):
        img = np.asarray(img)
        # Convert to uint8 if using float images.
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # Convert from [channel, height, width] to [height, width, channel].
        return einops.rearrange(img, "c h w -> h w c")

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
    return data


def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False, action_dim: int = 14) -> np.ndarray:
    if adapt_to_pi:
        if action_dim == 7:
            # Standard single-arm Aloha: 6 joints + 1 gripper
            state = _single_arm_joint_flip_mask() * state
            state[6] = _gripper_to_angular(state[6])
        elif action_dim == 14:
            # Standard Aloha
            state = _joint_flip_mask() * state
            # Reverse the gripper transformation that is being applied by the Aloha runtime.
            state[[6, 13]] = _gripper_to_angular(state[[6, 13]])
        elif action_dim == 16:
            # Franka dual-arm: 7 joints + 1 gripper per arm = 16 dims
            # Indices: [0-6] left joints, [7] left gripper, [8-14] right joints, [15] right gripper
            gripper_indices = [7, 15]
            # Flip joints (indices 0-6 and 8-14)
            joint_indices = [i for i in range(action_dim) if i not in gripper_indices]
            state[joint_indices] = -state[joint_indices]
            # Reverse the gripper transformation
            state[gripper_indices] = _gripper_to_angular(state[gripper_indices])
    return state


def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False, action_dim: int = 14) -> np.ndarray:
    if adapt_to_pi:
        if action_dim == 7:
            # Standard single-arm Aloha: flip joints, transform gripper
            actions = _single_arm_joint_flip_mask() * actions
            actions[:, [6]] = _gripper_from_angular(actions[:, [6]])
        elif action_dim == 14:
            # Standard Aloha: flip joints, transform grippers
            actions = _joint_flip_mask() * actions
            actions[:, [6, 13]] = _gripper_from_angular(actions[:, [6, 13]])
        elif action_dim == 16:
            # Franka dual-arm: 7 joints + 1 gripper per arm = 16 dims
            # Indices: [0-6] left joints, [7] left gripper, [8-14] right joints, [15] right gripper
            gripper_indices = [7, 15]
            # Flip joints (indices 0-6 and 8-14)
            joint_indices = [i for i in range(action_dim) if i not in gripper_indices]
            actions[:, joint_indices] = -actions[:, joint_indices]
            # Convert grippers from angular to linear space
            actions[:, gripper_indices] = _gripper_from_angular(actions[:, gripper_indices])
    return actions
