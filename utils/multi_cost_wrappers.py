# Copyright 2026 Mehrdad Moghimi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multi-dimensional cost wrappers for Safety Gymnasium.

Safety Gymnasium environments always return a *scalar* cost: the builder sums
every constraint violation into ``cost_sum`` and returns that single number.
The individual terms are still computed (``cost_hazards``, ``cost_vases_contact``,
``cost_gremlins``, ...) and forwarded through ``info``.

:class:`MultiCostWrapper` turns that scalar interface into a vector one: it
returns ``cost`` as a ``(num_costs,)`` array so downstream code can treat each
constraint as an independent budget. Components can be

* **native** — the per-obstacle terms the task already computes,
* **synthetic** — derived signals such as a speed limit or a control-effort
  limit, which make it possible to run multi-constraint experiments on tasks
  that only expose one native cost (e.g. the ``*Velocity-v1`` family or
  ``SafetyPointGoal1-v0``).

:class:`MultiCostAugmentedObservation` is the vector counterpart of
:class:`utils.safety_wrappers.AugmentedObservation`: it appends one cost stock
per cost dimension to the observation.

Example::

    from functools import partial
    import safety_gymnasium as gym
    from utils.multi_cost_wrappers import MultiCostWrapper, MultiCostAugmentedObservation

    envs = gym.vector.make(
        "SafetyPointButton1-v0",
        num_envs=8,
        wrappers=[
            partial(gym.wrappers.SafeRescaleAction, min_action=-1.0, max_action=1.0),
            partial(MultiCostWrapper, cost_specs=["auto"]),
            partial(MultiCostAugmentedObservation, cost_stock_min=-30.0),
        ],
        asynchronous=True,  # required: the sync vector env cannot stack vector costs
    )
    envs.get_attr("cost_names")[0]
    # ['buttons', 'gremlins', 'hazards']

Note:
    ``safety_gymnasium.vector.SafetySyncVectorEnv`` preallocates a scalar cost
    buffer and therefore cannot hold vector costs. Use ``asynchronous=True``
    (the async vector env stacks whatever the workers return) or a plain
    single environment.
"""

from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

# Keys produced by the builder that are not per-obstacle cost components.
_NON_COMPONENT_COST_KEYS = frozenset({"cost_sum", "cost_exception"})


def broadcast_to_costs(value, num_costs: int, name: str = "value") -> np.ndarray:
    """Broadcast a scalar / sequence to a ``(num_costs,)`` float32 array.

    Args:
        value: A scalar, a length-1 sequence (broadcast to every dimension) or a
            sequence of exactly ``num_costs`` entries.
        num_costs: Number of cost dimensions.
        name: Name used in the error message.

    Returns:
        Array of shape ``(num_costs,)``.
    """
    if value is None:
        raise ValueError(f"{name} must not be None")
    if np.isscalar(value):
        return np.full(num_costs, float(value), dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 1:
        return np.full(num_costs, float(arr[0]), dtype=np.float32)
    if arr.size != num_costs:
        raise ValueError(f"{name} has {arr.size} entries but the env exposes {num_costs} costs")
    return arr


class MultiCostWrapper(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Expose a vector of cost signals instead of Safety Gymnasium's scalar cost.

    Each entry of ``cost_specs`` describes one cost dimension:

    ==========================  ==================================================
    Spec                        Meaning
    ==========================  ==================================================
    ``"auto"``                  Expands to every native component the task
                                computes (``cost_hazards``, ``cost_vases_contact``,
                                ...), sorted by name. Falls back to ``"sum"`` when
                                the task exposes no decomposition (velocity envs).
    ``"sum"``                   The env's own scalar cost (``cost_sum``).
    ``"native:<key>"``          The native component ``info["cost_<key>"]``, e.g.
                                ``"native:hazards"``. Missing keys contribute 0.
    ``"speed:<limit>"``         1.0 when the agent's planar speed exceeds
                                ``limit``, else 0.0.
    ``"speed_excess:<limit>"``  ``max(0, speed - limit)`` (continuous variant).
    ``"ctrl:<limit>"``          1.0 when the control effort ``mean(action**2)``
                                exceeds ``limit``, else 0.0.
    ``"ctrl_excess:<limit>"``   ``max(0, mean(action**2) - limit)``.
    ``"dup:<scale>"``           ``scale`` times the env's scalar cost. Useful as a
                                sanity check: duplicated dimensions must behave
                                like the single-cost algorithm.
    ==========================  ==================================================

    The resolved layout is published as :attr:`cost_names` / :attr:`num_costs`
    so a training script can read it back with ``envs.get_attr("num_costs")[0]``.

    Args:
        env: The Safety Gymnasium environment to wrap.
        cost_specs: Component specifications (see table). Defaults to ``["auto"]``.
        cost_scales: Optional per-dimension multiplier applied to the final
            vector. Scalar or one value per resolved dimension.

    The environment's own scalar cost is preserved in ``info['native_cost']`` so a
    multi-cost run stays directly comparable to a single-cost baseline. When every
    dimension is native, ``cost_vector.sum() == native_cost``; adding synthetic
    dimensions makes the sum diverge from it, which is exactly why it is kept.

    Attributes:
        num_costs: Number of cost dimensions.
        cost_names: Human-readable name of each dimension.
    """

    def __init__(
        self,
        env: gym.Env,
        cost_specs: Sequence[str] | None = None,
        cost_scales=1.0,
    ):
        gym.utils.RecordConstructorArgs.__init__(
            self,
            cost_specs=cost_specs,
            cost_scales=cost_scales,
        )
        gym.Wrapper.__init__(self, env)

        specs = ["auto"] if cost_specs is None else list(cost_specs)

        resolved: list[str] = []
        for spec in specs:
            if spec == "auto":
                native = self._discover_native_cost_keys()
                if native:
                    resolved.extend(f"native:{key[len('cost_'):]}" for key in native)
                else:
                    resolved.append("sum")
            else:
                resolved.append(spec)

        if not resolved:
            raise ValueError("cost_specs resolved to an empty list of cost dimensions")

        self.cost_specs = resolved
        self._parsed = [self._parse_spec(spec) for spec in resolved]
        self.cost_names = [name for _, name, _ in self._parsed]
        self.num_costs = len(self._parsed)
        self.cost_scales = broadcast_to_costs(cost_scales, self.num_costs, "cost_scales")

        # Last computed cost vector, handy for debugging / rendering.
        self.cost_vector = np.zeros(self.num_costs, dtype=np.float32)

    # ------------------------------------------------------------------ setup

    def _discover_native_cost_keys(self) -> list[str]:
        """Return the sorted native ``cost_*`` keys the wrapped task computes.

        Safety Gymnasium only reveals the decomposition once MuJoCo has been
        stepped, so the components are probed with one throw-away reset. Tasks
        without a decomposition (the ``*Velocity-v1`` family) yield an empty list.
        """
        try:
            self.env.reset()
            cost = self.env.unwrapped.task.calculate_cost()
        except Exception:  # noqa: BLE001 - any failure just means "no decomposition"
            return []
        return sorted(
            key
            for key, value in cost.items()
            if key.startswith("cost_") and key not in _NON_COMPONENT_COST_KEYS
        )

    @staticmethod
    def _parse_spec(spec: str) -> tuple[str, str, float]:
        """Parse one spec string into ``(kind, display_name, parameter)``."""
        head, _, arg = spec.partition(":")

        if head == "sum":
            return "sum", "sum", 0.0
        if head == "native":
            if not arg:
                raise ValueError(f"'{spec}' must be written as 'native:<key>'")
            return "native", arg, 0.0
        if head in ("speed", "speed_excess", "ctrl", "ctrl_excess", "dup"):
            if not arg:
                raise ValueError(f"'{spec}' must be written as '{head}:<value>'")
            value = float(arg)
            return head, f"{head}{value:g}", value

        raise ValueError(
            f"Unknown cost spec '{spec}'. Expected one of: auto, sum, native:<key>, "
            "speed:<limit>, speed_excess:<limit>, ctrl:<limit>, ctrl_excess:<limit>, dup:<scale>"
        )

    # ------------------------------------------------------------- components

    def _agent_speed(self, info: dict) -> float:
        """Planar speed of the agent, for both navigation and velocity tasks."""
        if "x_velocity" in info:
            return float(np.hypot(info.get("x_velocity", 0.0), info.get("y_velocity", 0.0)))
        task = getattr(self.env.unwrapped, "task", None)
        agent = getattr(task, "agent", None)
        if agent is not None:
            return float(np.linalg.norm(np.asarray(agent.vel, dtype=np.float64)[:2]))
        return 0.0

    def _cost_vector(self, cost: float, action: np.ndarray, info: dict) -> np.ndarray:
        """Assemble the per-dimension cost vector for the current transition."""
        values = np.empty(self.num_costs, dtype=np.float32)
        speed = None
        effort = None

        for i, (kind, name, param) in enumerate(self._parsed):
            if kind == "sum":
                values[i] = float(cost)
            elif kind == "native":
                values[i] = float(info.get(f"cost_{name}", 0.0))
            elif kind == "dup":
                values[i] = param * float(cost)
            elif kind in ("speed", "speed_excess"):
                if speed is None:
                    speed = self._agent_speed(info)
                values[i] = float(speed > param) if kind == "speed" else max(0.0, speed - param)
            else:  # ctrl / ctrl_excess
                if effort is None:
                    effort = float(np.mean(np.square(np.asarray(action, dtype=np.float64))))
                values[i] = float(effort > param) if kind == "ctrl" else max(0.0, effort - param)

        return values * self.cost_scales

    # ------------------------------------------------------------------- api

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.cost_vector = np.zeros(self.num_costs, dtype=np.float32)
        return obs, info

    def step(self, action):
        """Step the env and replace the scalar cost with the cost vector."""
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        self.cost_vector = self._cost_vector(cost, action, info)
        # Keep the env's own scalar cost so multi-cost runs stay comparable to a
        # single-cost baseline even when synthetic dimensions are in the vector.
        info["native_cost"] = float(cost)
        return obs, reward, self.cost_vector, terminated, truncated, info


class MultiCostAugmentedObservation(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Augment observations with one cost stock per cost dimension.

    This is the vector counterpart of
    :class:`utils.safety_wrappers.AugmentedObservation`. For every cost
    dimension ``i`` the stock follows the same recursion as the scalar version:

        ``cost_stock_{t+1}[i] = (cost_stock_t[i] + cost_t[i] / cost_normalizer[i]) / gamma_cost``

    The stocks occupy the last :attr:`num_costs` entries of the observation,
    starting at :attr:`cost_stock_index` (``cost_stock_indices`` lists them all).

    Every per-dimension setting accepts either a scalar (broadcast to all
    dimensions) or one value per dimension.

    Args:
        env: Environment wrapped by :class:`MultiCostWrapper` (or any env with a
            scalar cost, which is treated as a single cost dimension).
        num_costs: Number of cost dimensions. Read from the wrapped env when None.
        add_cost_stock: If False the wrapper is a no-op on the observation.
        gamma_cost: Discount factor applied to the stock recursion (scalar).
        initial_cost_stock: Initial stock value(s), in raw cost units.
        cost_normalizer: Divisor applied to costs and to ``initial_cost_stock``
            before entering the observation.
        sample_initial_cost_stock: If True, draw each dimension's ``c0``
            independently from its bins on reset.
        cost_stock_min: Minimum ``c0`` per dimension for binned sampling.
        cost_stock_max: Maximum ``c0`` per dimension for binned sampling.
        contextual_lambda_bins: Number of ``c0`` bins, shared with the
            contextual Lagrangian.
    """

    def __init__(
        self,
        env: gym.Env,
        num_costs: int | None = None,
        add_cost_stock: bool = True,
        gamma_cost: float = 1.0,
        initial_cost_stock=0.0,
        cost_normalizer=10.0,
        sample_initial_cost_stock: bool = True,
        cost_stock_min=-30.0,
        cost_stock_max=0.0,
        contextual_lambda_bins: int = 31,
    ):
        gym.utils.RecordConstructorArgs.__init__(
            self,
            num_costs=num_costs,
            add_cost_stock=add_cost_stock,
            gamma_cost=gamma_cost,
            initial_cost_stock=initial_cost_stock,
            cost_normalizer=cost_normalizer,
            sample_initial_cost_stock=sample_initial_cost_stock,
            cost_stock_min=cost_stock_min,
            cost_stock_max=cost_stock_max,
            contextual_lambda_bins=contextual_lambda_bins,
        )
        gym.Wrapper.__init__(self, env)

        assert isinstance(env.observation_space, Box), "Observation space must be Box"
        assert env.observation_space.dtype in (np.float32, np.float64)

        if num_costs is None:
            num_costs = int(getattr(env, "num_costs", 1))
        self.num_costs = num_costs
        self.cost_names = list(getattr(env, "cost_names", [f"cost{i}" for i in range(num_costs)]))

        self.add_cost_stock = add_cost_stock
        self.sample_initial_cost_stock = sample_initial_cost_stock
        self.contextual_lambda_bins = contextual_lambda_bins

        def _broadcast(value, name):
            """Broadcast, naming the resolved cost layout when the length is wrong."""
            try:
                return broadcast_to_costs(value, num_costs, name)
            except ValueError as exc:
                raise ValueError(
                    f"{exc}. The environment resolved to the cost dimensions {self.cost_names}; "
                    f"per-dimension settings must have exactly {num_costs} entries (or be a scalar). "
                    f"Check that cost_specs and the per-dimension overrides agree."
                ) from None

        self.cost_normalizer = _broadcast(cost_normalizer, "cost_normalizer")
        self.cost_stock_min = _broadcast(cost_stock_min, "cost_stock_min")
        self.cost_stock_max = _broadcast(cost_stock_max, "cost_stock_max")
        self.gamma_cost = gamma_cost

        # Normalized initial stock (the value that actually enters the observation).
        self.initial_cost_stock = (
            _broadcast(initial_cost_stock, "initial_cost_stock") / self.cost_normalizer
        )
        self.cost_stock = self.initial_cost_stock.copy()

        # Discrete initial-stock choices, shared with the binned contextual Lagrangian.
        if self.sample_initial_cost_stock:
            self.bin_centers = np.stack(
                [
                    np.linspace(self.cost_stock_min[i], self.cost_stock_max[i], contextual_lambda_bins)
                    for i in range(num_costs)
                ]
            )  # (num_costs, contextual_lambda_bins)
        else:
            self.bin_centers = None

        self.cost_stock_index = None
        self.cost_stock_indices = []

        original_obs_dim = env.observation_space.shape[0]
        if self.add_cost_stock:
            self.cost_stock_index = original_obs_dim
            self.cost_stock_indices = list(range(original_obs_dim, original_obs_dim + num_costs))

        # Determine max timesteps from spec or TimeLimit wrapper (used only for info logging).
        if env.spec is not None and env.spec.max_episode_steps is not None:
            self.max_timesteps = env.spec.max_episode_steps
        else:
            wrapped_env = env
            while isinstance(wrapped_env, gym.Wrapper):
                if isinstance(wrapped_env, gym.wrappers.TimeLimit):
                    self.max_timesteps = wrapped_env._max_episode_steps
                    break
                wrapped_env = wrapped_env.env

            if not hasattr(self, "max_timesteps"):
                self.max_timesteps = 1000

        self.timesteps = 0

        low_list = list(self.observation_space.low)
        high_list = list(self.observation_space.high)
        if self.add_cost_stock:
            low_list.extend([-np.inf] * num_costs)
            high_list.extend([np.inf] * num_costs)

        self.observation_space = Box(
            np.array(low_list, dtype=np.float32),
            np.array(high_list, dtype=np.float32),
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Append the cost stocks (if enabled) to the observation."""
        if self.add_cost_stock:
            return np.concatenate([observation, self.cost_stock]).astype(self.observation_space.dtype)
        return observation.astype(self.observation_space.dtype)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment and reinitialize every cost stock.

        ``options['initial_cost_stock']`` (scalar or one value per dimension)
        overrides the per-episode stocks, which is what evaluation uses.
        """
        obs, info = self.env.reset(seed=seed, options=options)

        self.timesteps = 0

        if self.add_cost_stock:
            if options is not None and "initial_cost_stock" in options:
                override = broadcast_to_costs(
                    options["initial_cost_stock"], self.num_costs, "options['initial_cost_stock']"
                )
                self.cost_stock = override / self.cost_normalizer
            elif self.sample_initial_cost_stock:
                rng = self.np_random if getattr(self, "np_random", None) is not None else np.random.default_rng()
                # Each dimension draws its own budget independently.
                sampled = np.array(
                    [rng.choice(self.bin_centers[i]) for i in range(self.num_costs)], dtype=np.float32
                )
                self.cost_stock = sampled / self.cost_normalizer
            else:
                self.cost_stock = self.initial_cost_stock.copy()

        info["timestep"] = self.timesteps
        info["max_timesteps"] = self.max_timesteps

        return self.observation(obs), info

    def step(self, action):
        """Execute the action and update every cost stock."""
        obs, reward, cost, terminated, truncated, info = self.env.step(action)

        self.timesteps += 1

        cost = np.asarray(cost, dtype=np.float32).reshape(-1)
        if cost.size != self.num_costs:
            raise ValueError(
                f"Expected a cost vector of size {self.num_costs}, got {cost.size}. "
                "Wrap the environment with MultiCostWrapper first."
            )

        if self.add_cost_stock:
            self.cost_stock = (self.cost_stock + (cost / self.cost_normalizer)) / self.gamma_cost

        info["timestep"] = self.timesteps
        info["max_timesteps"] = self.max_timesteps

        return self.observation(obs), reward, cost, terminated, truncated, info


__all__ = [
    "MultiCostWrapper",
    "MultiCostAugmentedObservation",
    "broadcast_to_costs",
]
