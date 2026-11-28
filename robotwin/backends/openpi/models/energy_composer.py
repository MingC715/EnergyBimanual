"""JAX energy composition for building a bimanual policy from unimanual Pi0 models."""

import dataclasses
from typing import Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.shared import array_typing as at


@dataclasses.dataclass(frozen=True)
class EnergyComposedPi0Config(_model.BaseModelConfig):

    pi05: bool = True
    dtype: str = "bfloat16"
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    # OpenPI pi0/pi0.5 checkpoints use a padded 32D action/state interface.
    # Aloha/Franka semantic dimensions are represented by the fields below and
    # padded back to ``action_dim`` before entering the unimanual actors.
    action_dim: int = 32
    composed_action_dim: int = 16
    single_arm_action_dim: int = 8
    single_arm_model_action_dim: int = 32
    action_horizon: int = 50
    single_arm_max_token_len: int | None = None
    max_token_len: int | None = None
    def __post_init__(self):
        if self.single_arm_max_token_len is None:
            object.__setattr__(self, "single_arm_max_token_len", 200 if self.pi05 else 48)
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", self.single_arm_max_token_len)
        if self.composed_action_dim != 2 * self.single_arm_action_dim:
            raise ValueError(
                "EnergyComposedPi0Config currently expects exactly two arms: "
                f"composed_action_dim={self.composed_action_dim}, single_arm_action_dim={self.single_arm_action_dim}"
            )
        if self.action_dim < self.composed_action_dim:
            raise ValueError(
                f"action_dim ({self.action_dim}) must be >= composed_action_dim ({self.composed_action_dim})"
            )

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI05 if self.pi05 else _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "EnergyComposedPi0":
        return EnergyComposedPi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)
        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return observation_spec, action_spec

    def single_arm_config(self) -> pi0_config.Pi0Config:
        return pi0_config.Pi0Config(
            dtype=self.dtype,
            paligemma_variant=self.paligemma_variant,
            action_expert_variant=self.action_expert_variant,
            action_dim=self.single_arm_model_action_dim,
            action_horizon=self.action_horizon,
            max_token_len=self.single_arm_max_token_len,
            pi05=self.pi05,
        )


class EnergyComposedPi0(_model.BaseModel):
    """Compose two single-arm Pi0 or Pi0.5 policies."""

    def __init__(self, config: EnergyComposedPi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.config = config
        left_rng, right_rng = jax.random.split(rngs.params())
        single_cfg = config.single_arm_config()
        self.left_model = single_cfg.create(left_rng)
        self.right_model = single_cfg.create(right_rng)
        self.deterministic = True

    def _split_observation(self, observation: _model.Observation, arm: Literal["left", "right"]) -> _model.Observation:
        dim = self.config.single_arm_action_dim
        state = observation.state[..., :dim] if arm == "left" else observation.state[..., dim : 2 * dim]
        state = self._pad_last_dim(state, self.config.single_arm_model_action_dim)
        return dataclasses.replace(
            observation,
            state=state,
        )

    def _pad_last_dim(self, x: at.Array, target_dim: int) -> at.Array:
        pad = target_dim - x.shape[-1]
        if pad < 0:
            return x[..., :target_dim]
        if pad == 0:
            return x
        return jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad)])

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        left_rng, right_rng = jax.random.split(rng)
        dim = self.config.single_arm_action_dim
        left_noise = None if noise is None else self._pad_last_dim(noise[..., :dim], self.config.single_arm_model_action_dim)
        right_noise = (
            None if noise is None else self._pad_last_dim(noise[..., dim : 2 * dim], self.config.single_arm_model_action_dim)
        )
        left_actions = self.left_model.sample_actions(
            left_rng, self._split_observation(observation, "left"), num_steps=num_steps, noise=left_noise
        )
        right_actions = self.right_model.sample_actions(
            right_rng, self._split_observation(observation, "right"), num_steps=num_steps, noise=right_noise
        )
        composed_actions = jnp.concatenate(
            [
                left_actions[..., : self.config.single_arm_action_dim],
                right_actions[..., : self.config.single_arm_action_dim],
            ],
            axis=-1,
        )
        return self._pad_last_dim(composed_actions, self.config.action_dim)
