"""EGGROLL post-training of the deployed PPO actor's output layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import hyperscalees as hs
import jax
import jax.numpy as jnp
import numpy as np
import optax

from mjlab_microduck.eggroll.policy_io import DeployedPolicy

PyTree = Any


@dataclass(frozen=True)
class PostTrainingPolicyConfig:
    """The deliberately narrow v1 EGGROLL search contract."""

    sigma: float
    learning_rate: float
    rank: int = 1
    seed: int = 42
    noise_reuse: int = 1

    def __post_init__(self) -> None:
        if self.sigma <= 0.0 or self.learning_rate <= 0.0:
            raise ValueError("sigma and learning_rate must be positive")
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        if self.noise_reuse <= 0:
            raise ValueError("noise_reuse must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OutputLayerPolicy:
    """Frozen imported PPO feature extractor plus one trainable linear layer.

    Only the final 1,806 parameters enter the EGGROLL perturbation and optimizer
    trees. The deployed normalizer and first three layers are immutable arrays.
    """

    NOISER = hs.noiser.eggroll.EggRoll

    def __init__(
        self, deployed: DeployedPolicy, config: PostTrainingPolicyConfig
    ) -> None:
        self.deployed = deployed
        self.config = config
        self.frozen_mean = jnp.asarray(deployed.normalizer_mean, dtype=jnp.float32)
        self.frozen_denominator = jnp.asarray(
            deployed.normalizer_denominator, dtype=jnp.float32
        )
        self.frozen_layers = tuple(
            (
                jnp.asarray(layer.weight, dtype=jnp.float32),
                jnp.asarray(layer.bias, dtype=jnp.float32),
            )
            for layer in deployed.layers[:-1]
        )
        self.params: PyTree = {
            "weight": jnp.asarray(deployed.output_weight, dtype=jnp.float32),
            "bias": jnp.asarray(deployed.output_bias, dtype=jnp.float32),
        }
        # JAX arrays are immutable; retaining this initial tree provides the exact
        # source comparator after ``self.params`` advances during post-training.
        self.source_params: PyTree = self.params
        root_key = jax.random.key(config.seed)
        weight_key, bias_key = jax.random.split(root_key)
        self.es_tree_key: PyTree = {"weight": weight_key, "bias": bias_key}
        # HyperscaleES classifications: matrix multiply for weight, ordinary
        # parameter noise for bias.
        self.es_map: PyTree = {"weight": 1, "bias": 0}
        frozen, state = self.NOISER.init_noiser(
            self.params,
            config.sigma,
            config.learning_rate,
            solver=optax.adam,
            rank=config.rank,
            noise_reuse=config.noise_reuse,
            group_size=0,
        )
        self.frozen_noiser_params = frozen
        self.noiser_params = state

        def features(observation: jax.Array) -> jax.Array:
            value = (observation - self.frozen_mean) / self.frozen_denominator
            for weight, bias in self.frozen_layers:
                value = jax.nn.elu(value @ weight.T + bias)
            return value

        def candidate_forward(
            params: PyTree,
            noiser_params: PyTree,
            iterinfo: tuple[jax.Array, jax.Array],
            observation: jax.Array,
        ) -> jax.Array:
            hidden = features(observation)
            value = self.NOISER.do_mm(
                self.frozen_noiser_params,
                noiser_params,
                params["weight"],
                self.es_tree_key["weight"],
                iterinfo,
                hidden,
            )
            bias = self.NOISER.get_noisy_standard(
                self.frozen_noiser_params,
                noiser_params,
                params["bias"],
                self.es_tree_key["bias"],
                iterinfo,
            )
            return value + bias

        self._candidate_actions = jax.jit(
            jax.vmap(candidate_forward, in_axes=(None, None, 0, 0))
        )

        def base_forward(params: PyTree, observation: jax.Array) -> jax.Array:
            hidden = features(observation)
            return hidden @ params["weight"].T + params["bias"]

        self._base_actions = jax.jit(jax.vmap(base_forward, in_axes=(None, 0)))

    @property
    def trainable_parameter_count(self) -> int:
        return int(sum(value.size for value in jax.tree.leaves(self.params)))

    def candidate_actions(
        self, observations: jax.Array, *, generation: int
    ) -> jax.Array:
        observations = jnp.asarray(observations, dtype=jnp.float32)
        population = int(observations.shape[0])
        if observations.shape != (population, 61):
            raise ValueError(
                f"Candidate observations must be [population, 61], got "
                f"{observations.shape}"
            )
        if population < 2 or population % 2:
            raise ValueError(
                "Candidate population must be a non-empty antithetic batch"
            )
        iterinfo = jnp.stack(
            (
                jnp.full(population, generation, dtype=jnp.int32),
                jnp.arange(population, dtype=jnp.int32),
            ),
            axis=1,
        )
        return self._candidate_actions(
            self.params, self.noiser_params, iterinfo, observations
        )

    def base_actions(self, observations: jax.Array) -> jax.Array:
        observations = jnp.asarray(observations, dtype=jnp.float32)
        if observations.ndim != 2 or observations.shape[1] != 61:
            raise ValueError(
                f"Base observations must be [batch, 61], got {observations.shape}"
            )
        return self._base_actions(self.params, observations)

    def source_actions(self, observations: jax.Array) -> jax.Array:
        """Evaluate the immutable deployed source after the adapted mean changes."""

        observations = jnp.asarray(observations, dtype=jnp.float32)
        if observations.ndim != 2 or observations.shape[1] != 61:
            raise ValueError(
                f"Source observations must be [batch, 61], got {observations.shape}"
            )
        return self._base_actions(self.source_params, observations)

    def update(self, raw_fitness: np.ndarray, *, generation: int) -> np.ndarray:
        fitness = jnp.asarray(raw_fitness, dtype=jnp.float32)
        population = int(fitness.shape[0])
        if fitness.shape != (population,) or population % 2 != 0:
            raise ValueError("Fitness must be a non-empty even population vector")
        if not bool(jnp.all(jnp.isfinite(fitness))):
            raise FloatingPointError("Fitness contains non-finite values")
        converted = self.NOISER.convert_fitnesses(
            self.frozen_noiser_params, self.noiser_params, fitness
        )
        iterinfo = (
            jnp.full(population, generation, dtype=jnp.int32),
            jnp.arange(population, dtype=jnp.int32),
        )
        self.noiser_params, self.params = self.NOISER.do_updates(
            self.frozen_noiser_params,
            self.noiser_params,
            self.params,
            self.es_tree_key,
            converted,
            iterinfo,
            self.es_map,
        )
        leaves = jax.tree.leaves(self.params)
        if not all(bool(jnp.all(jnp.isfinite(value))) for value in leaves):
            raise FloatingPointError("EGGROLL update produced non-finite parameters")
        return np.asarray(jax.device_get(converted), dtype=np.float32)

    def output_parameters(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(jax.device_get(self.params["weight"]), dtype=np.float32),
            np.asarray(jax.device_get(self.params["bias"]), dtype=np.float32),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "params": jax.device_get(self.params),
            "noiser_params": jax.device_get(self.noiser_params),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if set(state) != {"params", "noiser_params"}:
            raise ValueError("Post-training policy state has unexpected keys")
        params = jax.tree.map(lambda value: jnp.asarray(value), state["params"])
        expected = jax.tree.map(lambda value: value.shape, self.params)
        actual = jax.tree.map(lambda value: value.shape, params)
        if expected != actual:
            raise ValueError("Post-training parameter shapes do not match policy")
        noiser = jax.tree.map(lambda value: jnp.asarray(value), state["noiser_params"])
        if jax.tree.structure(self.noiser_params) != jax.tree.structure(noiser):
            raise ValueError("Post-training optimizer state structure does not match")
        # HyperscaleES keeps ``sigma`` as a Python float alongside array-valued
        # Optax state. ``np.shape`` validates both scalar and array leaves.
        expected_shapes = jax.tree.map(np.shape, self.noiser_params)
        actual_shapes = jax.tree.map(np.shape, noiser)
        if expected_shapes != actual_shapes:
            raise ValueError("Post-training optimizer state shapes do not match")
        self.params = params
        self.noiser_params = noiser
