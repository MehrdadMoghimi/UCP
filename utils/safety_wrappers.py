import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from typing import Any


class AugmentedObservation(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """
    Wrapper that augments observations with an optional cost stock feature.

    The wrapper exposes ``cost_stock_index`` (the position of the cost stock in
    the augmented observation, or ``None`` when disabled) and tracks the
    episode timestep in ``info['timestep']`` / ``info['max_timesteps']``.

    Stock update equation (when ``add_cost_stock=True``):

        cost_stock_{t+1} = (cost_stock_t + cost_t / cost_normalizer) / gamma_cost

    Args:
        env: The Safety Gymnasium environment to wrap.
        add_cost_stock: If True, append the cost stock to observations.
        gamma_cost: Discount factor applied to the cost stock recursion.
        initial_cost_stock: Initial value for the cost stock.
        cost_normalizer: Divisor applied to instantaneous costs and to
            ``initial_cost_stock`` before being added to the observation.
        sample_initial_cost_stock: If True, sample ``c0`` from binned values
            on reset (matches the bins used by the contextual Lagrangian).
        cost_stock_min: Minimum ``c0`` value for binned initial-stock sampling.
        cost_stock_max: Maximum ``c0`` value for binned initial-stock sampling.
        contextual_lambda_bins: Number of ``c0`` bins shared with the
            contextual Lagrangian.
    """

    def __init__(
        self,
        env: gym.Env,
        add_cost_stock: bool = True,
        gamma_cost: float = 1.0,
        initial_cost_stock: float = 0.0,
        cost_normalizer: float = 10.0,
        sample_initial_cost_stock: bool = True,
        cost_stock_min: float = -30.0,
        cost_stock_max: float = 0.0,
        contextual_lambda_bins: int = 31,
    ):
        gym.utils.RecordConstructorArgs.__init__(
            self,
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
        assert env.observation_space.dtype == np.float32 or env.observation_space.dtype == np.float64

        self.add_cost_stock = add_cost_stock

        self.sample_initial_cost_stock = sample_initial_cost_stock
        self.cost_stock_min = cost_stock_min
        self.cost_stock_max = cost_stock_max
        self.contextual_lambda_bins = contextual_lambda_bins

        # Discrete initial-stock choices used by the binned contextual Lagrangian.
        if self.sample_initial_cost_stock:
            if self.cost_stock_min is None or self.cost_stock_max is None:
                raise ValueError("sample_initial_cost_stock requires cost_stock_min and cost_stock_max")
            self.bin_centers = np.linspace(self.cost_stock_min, self.cost_stock_max, self.contextual_lambda_bins)
        else:
            self.bin_centers = None

        self.cost_stock_index = None

        original_obs_dim = env.observation_space.shape[0]
        current_index = original_obs_dim

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

            if not hasattr(self, 'max_timesteps'):
                self.max_timesteps = 1000

        self.timesteps = 0

        if self.add_cost_stock:
            self.gamma_cost = gamma_cost
            self.cost_normalizer = cost_normalizer
            self.initial_cost_stock = initial_cost_stock / self.cost_normalizer
            self.cost_stock = self.initial_cost_stock
            self.cost_stock_index = current_index
            current_index += 1

        low_list = list(self.observation_space.low)
        high_list = list(self.observation_space.high)

        if self.add_cost_stock:
            low_list.append(-np.inf)
            high_list.append(np.inf)

        self.observation_space = Box(
            np.array(low_list, dtype=np.float32),
            np.array(high_list, dtype=np.float32),
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Append the cost stock (if enabled) to the observation."""
        if self.add_cost_stock:
            return np.append(observation, [self.cost_stock]).astype(self.observation_space.dtype)
        return observation.astype(self.observation_space.dtype)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment and reinitialize the cost stock.

        If ``options['initial_cost_stock']`` is provided it overrides the
        per-episode cost stock (used during evaluation). Otherwise, when
        ``sample_initial_cost_stock`` is True the stock is drawn from the
        configured bins; otherwise the fixed ``initial_cost_stock`` is used.
        """
        obs, info = super().reset(seed=seed, options=options)

        self.timesteps = 0

        if self.add_cost_stock:
            if options is not None and 'initial_cost_stock' in options:
                self.cost_stock = options['initial_cost_stock'] / self.cost_normalizer
            elif self.sample_initial_cost_stock:
                rng = self.np_random if hasattr(self, 'np_random') and self.np_random is not None else np.random.default_rng()
                sampled_stock = rng.choice(self.bin_centers)
                self.cost_stock = sampled_stock / self.cost_normalizer
            else:
                self.cost_stock = self.initial_cost_stock

        info['timestep'] = self.timesteps
        info['max_timesteps'] = self.max_timesteps

        return self.observation(obs), info

    def step(self, action):
        """Execute action and update the cost stock.

        Safety Gymnasium returns ``(obs, reward, cost, terminated, truncated, info)``.
        """
        obs, reward, cost, terminated, truncated, info = self.env.step(action)

        self.timesteps += 1

        if self.add_cost_stock:
            self.cost_stock = (self.cost_stock + (cost / self.cost_normalizer)) / self.gamma_cost

        info['timestep'] = self.timesteps
        info['max_timesteps'] = self.max_timesteps

        return self.observation(obs), reward, cost, terminated, truncated, info
