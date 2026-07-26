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

"""Evaluation utilities for trained multi-cost UCP models.

Mirrors :mod:`utils.evaluator` but keeps every cost dimension separate.
``mean_cost`` stays the total across dimensions (so existing plotting code keeps
working) and ``mean_cost_per_dim`` holds the breakdown, ordered like
``cost_names``.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import safety_gymnasium as gym
import torch
import tqdm

from utils.multi_cost_wrappers import MultiCostAugmentedObservation, MultiCostWrapper


def _load_evaluation_cache(cache_path: Path, eval_params: dict) -> dict:
    """Load evaluation results from cache if the parameters match."""
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "r") as f:
            cache_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"⚠ Cache file corrupted, will re-evaluate: {e}")
        return None

    if cache_data.get("eval_params", {}) != eval_params:
        print(f"⚠ Cache parameters mismatch, will re-evaluate")
        return None

    print(f"✓ Loading cached evaluation results from: {cache_path}")
    return cache_data["results"]


def _save_evaluation_cache(cache_path: Path, results: dict, eval_params: dict):
    """Write evaluation results next to the model."""
    with open(cache_path, "w") as f:
        json.dump({"eval_params": eval_params, "results": results}, f, indent=2)
    print(f"✓ Evaluation results cached to: {cache_path}")


def _build_env_and_actor(model_path: str, sample_initial_cost_stock: bool = None):
    """Rebuild the evaluation environment and the trained actor from a checkpoint.

    ``sample_initial_cost_stock=None`` keeps whatever the training run used; the
    multi-stock sweep passes ``False`` so ``reset(options=...)`` sets the budget.
    """
    import inspect

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    saved_args = checkpoint["args"]

    from ucp_multicost import Actor, Args

    # Filter saved_args to only include valid parameters for Args class
    # This handles added/removed args between versions
    valid_params = set(inspect.signature(Args).parameters.keys())
    args = Args(**{k: v for k, v in saved_args.items() if k in valid_params})

    if sample_initial_cost_stock is None:
        sample_initial_cost_stock = args.sample_initial_cost_stock

    device = torch.device("cpu")

    env = gym.make(args.env_id)
    env = gym.wrappers.SafeRescaleAction(env, min_action=-1.0, max_action=1.0)
    env = MultiCostWrapper(env, cost_specs=args.cost_specs)
    num_costs = env.num_costs
    cost_names = list(env.cost_names)

    saved_names = checkpoint.get("cost_names")
    if saved_names is not None and list(saved_names) != cost_names:
        raise ValueError(
            f"Cost layout mismatch: checkpoint was trained on {list(saved_names)} "
            f"but the environment now exposes {cost_names}"
        )

    env = MultiCostAugmentedObservation(
        env,
        num_costs=num_costs,
        add_cost_stock=args.add_cost_stock,
        gamma_cost=args.gamma_cost,
        initial_cost_stock=args.initial_cost_stocks if args.initial_cost_stocks is not None else args.initial_cost_stock,
        cost_normalizer=args.cost_normalizers if args.cost_normalizers is not None else args.cost_normalizer,
        sample_initial_cost_stock=sample_initial_cost_stock,
        cost_stock_min=args.cost_stock_mins if args.cost_stock_mins is not None else args.cost_stock_min,
        cost_stock_max=args.cost_stock_maxs if args.cost_stock_maxs is not None else args.cost_stock_max,
        contextual_lambda_bins=args.contextual_lambda_bins,
    )

    n_act = math.prod(env.action_space.shape)
    n_obs = math.prod(env.observation_space.shape)

    n_augmented_features = num_costs if args.add_cost_stock else 0
    n_base_obs = n_obs - n_augmented_features

    actor = Actor(
        env.action_space,
        n_obs=n_obs,
        n_act=n_act,
        device=device,
        hidden_sizes=args.actor_hidden_sizes,
        use_embedding=args.use_embedding,
        embed_layers=args.embed_layers,
        embed_connection_type=args.embed_connection_type,
        embed_connection_layer=args.embed_connection_layer,
        n_base_obs=n_base_obs,
        n_aug_features=n_augmented_features,
    ).to(device)

    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()

    return env, actor, args, cost_names, num_costs, device


def _run_episodes(env, actor, device, gamma, num_costs, episodes, temperature, desc, reset_options=None):
    """Roll out ``episodes`` episodes and collect per-dimension and combined cost statistics."""
    returns, costs, discounted_returns, discounted_costs = [], [], [], []
    native_costs, discounted_native_costs = [], []

    for _ in tqdm.tqdm(range(episodes), desc=desc):
        obs, _ = env.reset(options=reset_options)
        obs = torch.Tensor(obs).unsqueeze(0).to(device)  # Add batch dimension

        episode_return = 0.0
        episode_cost = np.zeros(num_costs, dtype=np.float64)
        episode_native_cost = 0.0
        discounted_return = 0.0
        discounted_cost = np.zeros(num_costs, dtype=np.float64)
        discounted_native_cost = 0.0
        discount = 1.0  # gamma^t
        terminated = False
        truncated = False

        while not (terminated or truncated):
            with torch.no_grad():
                actions, _, _, _, _, _ = actor.get_action(obs, temperature=temperature)

            # Remove batch dimension for single env step (already on CPU)
            action = actions.numpy()[0]
            next_obs, reward, cost, terminated, truncated, info = env.step(action)
            # The env's own scalar cost, i.e. what a single-cost baseline reports
            native_cost = float(info.get("native_cost", np.sum(cost)))

            episode_return += reward
            episode_cost += cost
            episode_native_cost += native_cost
            discounted_return += discount * reward
            discounted_cost += discount * cost
            discounted_native_cost += discount * native_cost
            discount *= gamma

            obs = torch.Tensor(next_obs).unsqueeze(0).to(device)

        returns.append(float(episode_return))
        costs.append(episode_cost)
        native_costs.append(episode_native_cost)
        discounted_returns.append(float(discounted_return))
        discounted_costs.append(discounted_cost)
        discounted_native_costs.append(discounted_native_cost)

    return (returns, np.stack(costs), np.asarray(native_costs),
            discounted_returns, np.stack(discounted_costs), np.asarray(discounted_native_costs))


def _summarize(returns, costs, native_costs, discounted_returns, discounted_costs,
               discounted_native_costs) -> dict:
    """Build the result dict.

    Three views of the same episodes:

    * ``mean_cost`` / ``std_cost`` — the summed multi-cost vector,
    * ``mean_cost_per_dim`` — the per-constraint breakdown,
    * ``mean_native_cost`` — the environment's own scalar cost, the number a
      single-cost run reports. Identical to ``mean_cost`` when every dimension is
      native; it differs as soon as synthetic dimensions are in play.
    """
    total_costs = costs.sum(axis=1)
    total_discounted_costs = discounted_costs.sum(axis=1)
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_cost": float(np.mean(total_costs)),
        "std_cost": float(np.std(total_costs)),
        "mean_cost_per_dim": [float(v) for v in costs.mean(axis=0)],
        "std_cost_per_dim": [float(v) for v in costs.std(axis=0)],
        "mean_native_cost": float(np.mean(native_costs)),
        "std_native_cost": float(np.std(native_costs)),
        "mean_discounted_return": float(np.mean(discounted_returns)),
        "std_discounted_return": float(np.std(discounted_returns)),
        "mean_discounted_cost": float(np.mean(total_discounted_costs)),
        "std_discounted_cost": float(np.std(total_discounted_costs)),
        "mean_discounted_cost_per_dim": [float(v) for v in discounted_costs.mean(axis=0)],
        "mean_discounted_native_cost": float(np.mean(discounted_native_costs)),
        "returns": [float(r) for r in returns],
        "costs": [[float(v) for v in row] for row in costs],
        "native_costs": [float(v) for v in native_costs],
        "discounted_returns": [float(r) for r in discounted_returns],
        "discounted_costs": [[float(v) for v in row] for row in discounted_costs],
        "discounted_native_costs": [float(v) for v in discounted_native_costs],
    }


def evaluate_ucp_multicost(
    model_path: str,
    eval_episodes: int,
    evaluation_temperature: float = 1.0,
    use_cache: bool = True,
):
    """
    Evaluate a trained multi-cost UCP model.

    Args:
        model_path: Path to the saved model checkpoint (.pt file)
        eval_episodes: Number of episodes to evaluate
        evaluation_temperature: Temperature parameter for policy (1.0=stochastic, 0.0=deterministic)
        use_cache: Whether to use cached results if available (default: True)

    Returns:
        Dictionary with ``mean_return`` / ``mean_cost`` (total across constraints),
        ``mean_cost_per_dim`` (one entry per constraint), their discounted and std
        counterparts, the raw per-episode lists, ``cost_names`` and ``gamma``.
    """
    eval_params = {
        "eval_episodes": eval_episodes,
        "evaluation_temperature": evaluation_temperature,
        "multi_cost": True,
    }
    param_hash = hashlib.md5(
        f"episodes_{eval_episodes}_temp_{evaluation_temperature}".encode()
    ).hexdigest()[:8]
    model_path_obj = Path(model_path)
    cache_path = model_path_obj.parent / f"{model_path_obj.stem}_eval_multicost_{param_hash}.json"

    if use_cache:
        cached_results = _load_evaluation_cache(cache_path, eval_params)
        if cached_results is not None:
            return cached_results
        print(f"⚠ Cache at {cache_path} not found or invalid, proceeding with evaluation")

    # Keep the training-time budget distribution for the plain evaluation.
    env, actor, args, cost_names, num_costs, device = _build_env_and_actor(model_path)

    print(f"\n{'='*50}")
    print(f"Starting evaluation for {eval_episodes} episodes...")
    print(f"Model: {model_path}")
    print(f"Cost dimensions: {cost_names}")
    print(f"{'='*50}\n")

    episode_stats = _run_episodes(
        env, actor, device, args.gamma, num_costs, eval_episodes, evaluation_temperature, "Evaluating"
    )
    env.close()

    results = _summarize(*episode_stats)
    results["cost_names"] = cost_names
    results["gamma"] = float(args.gamma)

    print(f"\n{'='*50}")
    print(f"Evaluation Results:")
    print(f"  Mean Return: {results['mean_return']:.2f} ± {results['std_return']:.2f}")
    print(f"  Mean Cost (combined, native env cost): "
          f"{results['mean_native_cost']:.2f} ± {results['std_native_cost']:.2f}")
    print(f"  Mean Cost (sum of dimensions): {results['mean_cost']:.2f} ± {results['std_cost']:.2f}")
    for name, value, std in zip(cost_names, results["mean_cost_per_dim"], results["std_cost_per_dim"]):
        print(f"    {name}: {value:.2f} ± {std:.2f}")
    print(f"  Mean Discounted Return (γ={args.gamma:.3f}): "
          f"{results['mean_discounted_return']:.2f} ± {results['std_discounted_return']:.2f}")
    print(f"{'='*50}\n")

    _save_evaluation_cache(cache_path, results, eval_params)
    return results


def evaluate_ucp_multicost_multi_stock(
    model_path: str,
    eval_episodes_per_stock: int,
    initial_cost_stocks: list,
    evaluation_temperature: float = 0.0,
    use_cache: bool = True,
    cache_dir: str = None,
):
    """
    Evaluate a trained multi-cost UCP model across multiple initial cost budgets.

    Each value in ``initial_cost_stocks`` is applied to *every* cost dimension, so
    the sweep answers "how does the agent behave when all budgets are tight/loose".
    Pass a list of lists to give each dimension its own budget per sweep point.

    Args:
        model_path: Path to the saved model checkpoint (.pt file)
        eval_episodes_per_stock: Number of episodes to evaluate per initial stock value
        initial_cost_stocks: Budgets to sweep (scalars broadcast to all dimensions,
            or one list per sweep point with a value per dimension)
        evaluation_temperature: Temperature parameter for policy (1.0=stochastic, 0.0=deterministic)
        use_cache: Whether to use cached results if available (default: True)
        cache_dir: Optional directory to store/read the evaluation JSON instead of next to the model.

    Returns:
        Dictionary with ``per_stock_results`` (keyed by budget), ``aggregate``,
        ``initial_cost_stocks``, ``cost_names`` and ``gamma``.
    """
    eval_params = {
        "eval_episodes_per_stock": eval_episodes_per_stock,
        "initial_cost_stocks": [
            sorted(np.atleast_1d(s).tolist()) if not np.isscalar(s) else float(s)
            for s in initial_cost_stocks
        ],
        "evaluation_temperature": evaluation_temperature,
        "multi_stock": True,
        "multi_cost": True,
    }

    stock_str = "_".join(f"{np.mean(s):.1f}" for s in initial_cost_stocks)
    param_hash = hashlib.md5(
        f"multi_stock_{stock_str}_eps{eval_episodes_per_stock}_temp_{evaluation_temperature}".encode()
    ).hexdigest()[:8]
    model_path_obj = Path(model_path)
    _cache_dir = Path(cache_dir) if cache_dir is not None else model_path_obj.parent
    _cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_dir / f"{model_path_obj.stem}_eval_multicost_multistock_{param_hash}.json"

    if use_cache:
        cached_results = _load_evaluation_cache(cache_path, eval_params)
        if cached_results is not None:
            return cached_results

    # Disable random initial-stock sampling so each episode honours the per-call options override.
    env, actor, args, cost_names, num_costs, device = _build_env_and_actor(
        model_path, sample_initial_cost_stock=False
    )

    print(f"\n{'='*70}")
    print(f"Multi-Stock Evaluation (multi-cost)")
    print(f"Model: {model_path}")
    print(f"Cost dimensions: {cost_names}")
    print(f"Initial cost stocks: {initial_cost_stocks}")
    print(f"Episodes per stock: {eval_episodes_per_stock}")
    print(f"{'='*70}\n")

    per_stock_results = {}
    all_returns, all_costs, all_native_costs = [], [], []
    all_discounted_returns, all_discounted_costs, all_discounted_native_costs = [], [], []

    for initial_stock in initial_cost_stocks:
        print(f"\nEvaluating with initial_cost_stock = {initial_stock}")

        (returns, costs, native_costs,
         discounted_returns, discounted_costs, discounted_native_costs) = _run_episodes(
            env, actor, device, args.gamma, num_costs, eval_episodes_per_stock,
            evaluation_temperature, f"Stock={initial_stock}",
            reset_options={"initial_cost_stock": initial_stock},
        )

        stock_results = _summarize(returns, costs, native_costs,
                                   discounted_returns, discounted_costs, discounted_native_costs)
        # Scalar sweeps keep float keys (matching the single-cost evaluator);
        # per-dimension sweeps are keyed by their mean budget.
        per_stock_results[float(np.mean(initial_stock))] = stock_results

        all_returns.extend(returns)
        all_costs.append(costs)
        all_native_costs.extend(native_costs)
        all_discounted_returns.extend(discounted_returns)
        all_discounted_costs.append(discounted_costs)
        all_discounted_native_costs.extend(discounted_native_costs)

        print(f"  Return: {stock_results['mean_return']:.2f} ± {stock_results['std_return']:.2f}")
        print(f"  Cost (combined): {stock_results['mean_native_cost']:.2f} ± {stock_results['std_native_cost']:.2f}")
        print(f"  Cost (sum of dims): {stock_results['mean_cost']:.2f} ± {stock_results['std_cost']:.2f}")
        for name, value in zip(cost_names, stock_results["mean_cost_per_dim"]):
            print(f"    {name}: {value:.2f}")

    env.close()

    aggregate = _summarize(
        all_returns, np.concatenate(all_costs), np.asarray(all_native_costs),
        all_discounted_returns, np.concatenate(all_discounted_costs),
        np.asarray(all_discounted_native_costs),
    )
    # The aggregate does not need the full per-episode dump.
    for key in ("returns", "costs", "native_costs",
                "discounted_returns", "discounted_costs", "discounted_native_costs"):
        aggregate.pop(key)
    aggregate["total_episodes"] = len(all_returns)

    print(f"\n{'='*70}")
    print(f"Multi-Stock Evaluation Summary:")
    print(f"{'='*70}")
    header = f"{'Initial Stock':>15} | {'Return':>20} | {'Cost (combined)':>20} | " + " | ".join(
        f"{name:>14}" for name in cost_names
    )
    print(header)
    print("-" * len(header))
    for stock in sorted(per_stock_results.keys()):
        res = per_stock_results[stock]
        per_dim = " | ".join(f"{v:>14.2f}" for v in res["mean_cost_per_dim"])
        print(f"{stock:>15.1f} | {res['mean_return']:>8.2f} ± {res['std_return']:<8.2f} | "
              f"{res['mean_native_cost']:>8.2f} ± {res['std_native_cost']:<8.2f} | {per_dim}")
    print("-" * len(header))
    agg_per_dim = " | ".join(f"{v:>14.2f}" for v in aggregate["mean_cost_per_dim"])
    print(f"{'Aggregate':>15} | {aggregate['mean_return']:>8.2f} ± {aggregate['std_return']:<8.2f} | "
          f"{aggregate['mean_native_cost']:>8.2f} ± {aggregate['std_native_cost']:<8.2f} | {agg_per_dim}")
    print(f"{'='*70}\n")

    results = {
        "per_stock_results": per_stock_results,
        "aggregate": aggregate,
        "initial_cost_stocks": [
            float(s) if np.isscalar(s) else [float(v) for v in np.atleast_1d(s)]
            for s in initial_cost_stocks
        ],
        "cost_names": cost_names,
        "gamma": float(args.gamma),
    }

    _save_evaluation_cache(cache_path, results, eval_params)
    return results


__all__ = [
    "evaluate_ucp_multicost",
    "evaluate_ucp_multicost_multi_stock",
]
