"""
Evaluation utilities for trained RL models.
"""
import math
import os
import json
import hashlib
import numpy as np
import torch
import safety_gymnasium as gym
import tqdm
from pathlib import Path

from utils.safety_wrappers import AugmentedObservation


def _get_cache_path(model_path: str, eval_episodes: int, evaluation_temperature: float) -> Path:
    """
    Generate cache file path based on model path and evaluation parameters.
    
    Args:
        model_path: Path to the model checkpoint
        eval_episodes: Number of evaluation episodes
        evaluation_temperature: Temperature parameter
        
    Returns:
        Path to cache file
    """
    model_path = Path(model_path)
    model_dir = model_path.parent
    model_name = model_path.stem
    
    # Create a hash of the evaluation parameters
    param_str = f"episodes_{eval_episodes}_temp_{evaluation_temperature}"
    param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
    
    cache_filename = f"{model_name}_eval_{param_hash}.json"
    return model_dir / cache_filename


def _save_evaluation_cache(cache_path: Path, results: dict, eval_params: dict):
    """
    Save evaluation results to cache file.
    
    Args:
        cache_path: Path to save cache file
        results: Evaluation results dictionary
        eval_params: Evaluation parameters used
    """
    cache_data = {
        'eval_params': eval_params,
        'results': {
            'mean_return': float(results['mean_return']),
            'std_return': float(results['std_return']),
            'mean_cost': float(results['mean_cost']),
            'std_cost': float(results['std_cost']),
            'returns': [float(r) for r in results['returns']],
            'costs': [float(c) for c in results['costs']],
        }
    }
    
    # Add discounted metrics if available (for backward compatibility)
    if 'mean_discounted_return' in results:
        cache_data['results']['mean_discounted_return'] = float(results['mean_discounted_return'])
        cache_data['results']['std_discounted_return'] = float(results['std_discounted_return'])
        cache_data['results']['mean_discounted_cost'] = float(results['mean_discounted_cost'])
        cache_data['results']['std_discounted_cost'] = float(results['std_discounted_cost'])
        cache_data['results']['discounted_returns'] = [float(r) for r in results['discounted_returns']]
        cache_data['results']['discounted_costs'] = [float(c) for c in results['discounted_costs']]
        cache_data['results']['gamma'] = float(results['gamma'])
    
    with open(cache_path, 'w') as f:
        json.dump(cache_data, f, indent=2)
    
    print(f"✓ Evaluation results cached to: {cache_path}")


def _load_evaluation_cache(cache_path: Path, eval_params: dict) -> dict:
    """
    Load evaluation results from cache if parameters match.
    
    Args:
        cache_path: Path to cache file
        eval_params: Current evaluation parameters
        
    Returns:
        Cached results dictionary or None if cache is invalid
    """
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        
        # Verify parameters match
        cached_params = cache_data.get('eval_params', {})
        if cached_params == eval_params:
            print(f"✓ Loading cached evaluation results from: {cache_path}")
            results = cache_data['results']
            # Convert lists back to numpy arrays if needed (keep as lists for now)
            # For backward compatibility, add default values for discounted metrics if not present
            if 'mean_discounted_return' not in results:
                results['mean_discounted_return'] = None
                results['std_discounted_return'] = None
                results['mean_discounted_cost'] = None
                results['std_discounted_cost'] = None
                results['discounted_returns'] = None
                results['discounted_costs'] = None
                results['gamma'] = None
            return results
        else:
            print(f"⚠ Cache parameters mismatch, will re-evaluate")
            return None
            
    except (json.JSONDecodeError, KeyError) as e:
        print(f"⚠ Cache file corrupted, will re-evaluate: {e}")
        return None


def evaluate_ucp(model_path: str, eval_episodes: int, evaluation_temperature: float = 1.0, use_cache: bool = True):
    """
    Evaluate a trained UCP model with embedding support.
    
    Args:
        model_path: Path to the saved model checkpoint (.pt file)
        eval_episodes: Number of episodes to evaluate
        evaluation_temperature: Temperature parameter for policy (1.0=stochastic, 0.0=deterministic)
        use_cache: Whether to use cached results if available (default: True)
        
    Returns:
        Dictionary containing evaluation results:
            - mean_return: Mean episodic return
            - std_return: Standard deviation of returns
            - mean_cost: Mean episodic cost
            - std_cost: Standard deviation of costs
            - mean_discounted_return: Mean discounted return (using model's gamma)
            - std_discounted_return: Standard deviation of discounted returns
            - mean_discounted_cost: Mean discounted cost (using model's gamma)
            - std_discounted_cost: Standard deviation of discounted costs
            - returns: List of all episode returns
            - costs: List of all episode costs
            - discounted_returns: List of all discounted returns
            - discounted_costs: List of all discounted costs
            - gamma: Discount factor used
    """
    # Check cache first
    eval_params = {
        'eval_episodes': eval_episodes,
        'evaluation_temperature': evaluation_temperature,
    }
    
    if use_cache:
        cache_path = _get_cache_path(model_path, eval_episodes, evaluation_temperature)
        cached_results = _load_evaluation_cache(cache_path, eval_params)
        if cached_results is not None:
            print(f"✓ Loaded cached evaluation results from: {cache_path}")
            return cached_results
        else:
            print(f"⚠ Cache at {cache_path} not found or invalid, proceeding with evaluation")


    # Load checkpoint on CPU for faster evaluation
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    saved_args = checkpoint['args']
    
    from ucp import Actor, Args

    # Filter saved_args to only include valid parameters for Args class
    # This handles added/removed args between versions
    import inspect
    valid_params = set(inspect.signature(Args).parameters.keys())
    filtered_args = {k: v for k, v in saved_args.items() if k in valid_params}
    
    args = Args(**filtered_args)
    
    device = torch.device("cpu")
    
    env = gym.make(args.env_id)
    env = gym.wrappers.SafeRescaleAction(env, min_action=-1.0, max_action=1.0)
    
    if args.add_cost_stock:
        env = AugmentedObservation(
            env,
            add_cost_stock=args.add_cost_stock,
            gamma_cost=args.gamma_cost,
            initial_cost_stock=args.initial_cost_stock,
            cost_normalizer=args.cost_normalizer,
        )
    
    n_act = math.prod(env.action_space.shape)
    n_obs = math.prod(env.observation_space.shape)
    
    n_augmented_features = 1 if args.add_cost_stock else 0
    n_base_obs = n_obs - n_augmented_features
    
    # Reconstruct actor
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
    
    # Load actor weights
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()
    
    # Run evaluation
    print(f"\n{'='*50}")
    print(f"Starting evaluation for {eval_episodes} episodes...")
    print(f"Model: {model_path}")
    print(f"{'='*50}\n")
    
    eval_returns = []
    eval_costs = []
    eval_discounted_returns = []
    eval_discounted_costs = []
    
    gamma = args.gamma  # Use gamma from model configuration
    
    for eval_ep in tqdm.tqdm(range(eval_episodes), desc="Evaluating"):
        obs, _ = env.reset()
        obs = torch.Tensor(obs).unsqueeze(0).to(device)  # Add batch dimension
        episode_return = 0.0
        episode_cost = 0.0
        discounted_return = 0.0
        discounted_cost = 0.0
        discount = 1.0  # gamma^t
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            with torch.no_grad():
                actions, _, _, _, _, _ = actor.get_action(obs, temperature=evaluation_temperature)
            
            # Remove batch dimension for single env step (already on CPU)
            action = actions.numpy()[0]
            next_obs, reward, cost, terminated, truncated, info = env.step(action)
            
            episode_return += reward
            episode_cost += cost
            discounted_return += discount * reward
            discounted_cost += discount * cost
            discount *= gamma
            
            obs = torch.Tensor(next_obs).unsqueeze(0).to(device)
        
        eval_returns.append(episode_return)
        eval_costs.append(episode_cost)
        eval_discounted_returns.append(discounted_return)
        eval_discounted_costs.append(discounted_cost)
    
    mean_eval_return = np.mean(eval_returns)
    mean_eval_cost = np.mean(eval_costs)
    std_eval_return = np.std(eval_returns)
    std_eval_cost = np.std(eval_costs)
    mean_eval_discounted_return = np.mean(eval_discounted_returns)
    mean_eval_discounted_cost = np.mean(eval_discounted_costs)
    std_eval_discounted_return = np.std(eval_discounted_returns)
    std_eval_discounted_cost = np.std(eval_discounted_costs)
    
    print(f"\n{'='*50}")
    print(f"Evaluation Results:")
    print(f"  Mean Return: {mean_eval_return:.2f} ± {std_eval_return:.2f}")
    print(f"  Mean Cost: {mean_eval_cost:.2f} ± {std_eval_cost:.2f}")
    print(f"  Mean Discounted Return (γ={gamma:.3f}): {mean_eval_discounted_return:.2f} ± {std_eval_discounted_return:.2f}")
    print(f"  Mean Discounted Cost (γ={gamma:.3f}): {mean_eval_discounted_cost:.2f} ± {std_eval_discounted_cost:.2f}")
    print(f"{'='*50}\n")
    
    env.close()
    
    results = {
        'mean_return': mean_eval_return,
        'std_return': std_eval_return,
        'mean_cost': mean_eval_cost,
        'std_cost': std_eval_cost,
        'mean_discounted_return': mean_eval_discounted_return,
        'std_discounted_return': std_eval_discounted_return,
        'mean_discounted_cost': mean_eval_discounted_cost,
        'std_discounted_cost': std_eval_discounted_cost,
        'returns': eval_returns,
        'costs': eval_costs,
        'discounted_returns': eval_discounted_returns,
        'discounted_costs': eval_discounted_costs,
        'gamma': gamma,
    }
    
    # Always save to cache (use_cache only controls whether to load)
    cache_path = _get_cache_path(model_path, eval_episodes, evaluation_temperature)
    _save_evaluation_cache(cache_path, results, eval_params)
    
    return results


def evaluate_ucp_multi_stock(
    model_path: str,
    eval_episodes_per_stock: int,
    initial_cost_stocks: list,
    evaluation_temperature: float = 0.0,
    use_cache: bool = True,
    cache_dir: str = None,
):
    """
    Evaluate a trained UCP model across multiple initial cost stock values.
    
    This function evaluates the model with different initial cost stock values to understand
    how the agent performs under different "budget" conditions.
    
    Args:
        model_path: Path to the saved model checkpoint (.pt file)
        eval_episodes_per_stock: Number of episodes to evaluate per initial stock value
        initial_cost_stocks: List of initial cost stock values to evaluate (e.g., [-25, -20, -15, -10, -5, 0])
        evaluation_temperature: Temperature parameter for policy (1.0=stochastic, 0.0=deterministic)
        use_cache: Whether to use cached results if available (default: True)
        cache_dir: Optional directory to store/read the evaluation JSON instead of next to the model.
            The directory is created if it does not exist.
        
    Returns:
        Dictionary containing evaluation results per initial stock:
            - per_stock_results: Dict mapping initial_stock -> {mean_return, std_return, mean_cost, std_cost, ...}
            - aggregate: Overall aggregated results across all stocks
            - initial_cost_stocks: List of stock values used
    """
    from typing import List
    
    # Check cache first
    eval_params = {
        'eval_episodes_per_stock': eval_episodes_per_stock,
        'initial_cost_stocks': sorted(initial_cost_stocks),
        'evaluation_temperature': evaluation_temperature,
        'multi_stock': True
    }
    
    # Generate cache path with stock info
    stock_str = "_".join([f"{s:.1f}" for s in sorted(initial_cost_stocks)])
    param_str = f"multi_stock_{stock_str}_eps{eval_episodes_per_stock}_temp_{evaluation_temperature}"
    param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
    model_path_obj = Path(model_path)
    _cache_dir = Path(cache_dir) if cache_dir is not None else model_path_obj.parent
    _cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_dir / f"{model_path_obj.stem}_eval_multistock_{param_hash}.json"
    if use_cache:
        cached_results = _load_evaluation_cache(cache_path, eval_params)
        if cached_results is not None:
            print(f"✓ Loaded cached multi-stock evaluation results from: {cache_path}")
            return cached_results

    
    # Load checkpoint on CPU for faster evaluation
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    saved_args = checkpoint['args']
    
    from ucp import Actor, Args

    import inspect
    valid_params = set(inspect.signature(Args).parameters.keys())
    filtered_args = {k: v for k, v in saved_args.items() if k in valid_params}
    
    args = Args(**filtered_args)
    
    device = torch.device("cpu")
    
    env = gym.make(args.env_id)
    env = gym.wrappers.SafeRescaleAction(env, min_action=-1.0, max_action=1.0)
    
    # Disable random initial-stock sampling so each episode honours the per-call options override.
    if args.add_cost_stock:
        env = AugmentedObservation(
            env,
            add_cost_stock=args.add_cost_stock,
            gamma_cost=args.gamma_cost,
            initial_cost_stock=args.initial_cost_stock,  # Default, overridden via reset(options=...)
            cost_normalizer=args.cost_normalizer,
            sample_initial_cost_stock=False,
        )
    
    n_act = math.prod(env.action_space.shape)
    n_obs = math.prod(env.observation_space.shape)
    
    n_augmented_features = 1 if args.add_cost_stock else 0
    n_base_obs = n_obs - n_augmented_features
    
    # Reconstruct actor
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
    
    # Load actor weights
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()
    
    gamma = args.gamma  # Use gamma from model configuration
    
    # Run evaluation for each initial cost stock
    print(f"\n{'='*70}")
    print(f"Multi-Stock Evaluation")
    print(f"Model: {model_path}")
    print(f"Initial cost stocks: {initial_cost_stocks}")
    print(f"Episodes per stock: {eval_episodes_per_stock}")
    print(f"{'='*70}\n")
    
    per_stock_results = {}
    all_returns = []
    all_costs = []
    all_discounted_returns = []
    all_discounted_costs = []
    
    for initial_stock in initial_cost_stocks:
        print(f"\nEvaluating with initial_cost_stock = {initial_stock}")
        
        eval_returns = []
        eval_costs = []
        eval_discounted_returns = []
        eval_discounted_costs = []
        
        for eval_ep in tqdm.tqdm(range(eval_episodes_per_stock), desc=f"Stock={initial_stock}"):
            # Reset with specific initial cost stock via options
            obs, _ = env.reset(options={'initial_cost_stock': initial_stock})
            obs = torch.Tensor(obs).unsqueeze(0).to(device)  # Add batch dimension
            
            episode_return = 0.0
            episode_cost = 0.0
            discounted_return = 0.0
            discounted_cost = 0.0
            discount = 1.0  # gamma^t
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                with torch.no_grad():
                    actions, _, _, _, _, _ = actor.get_action(obs, temperature=evaluation_temperature)
                
                # Remove batch dimension for single env step (already on CPU)
                action = actions.numpy()[0]
                next_obs, reward, cost, terminated, truncated, info = env.step(action)
                
                episode_return += reward
                episode_cost += cost
                discounted_return += discount * reward
                discounted_cost += discount * cost
                discount *= gamma
                
                obs = torch.Tensor(next_obs).unsqueeze(0).to(device)
            
            eval_returns.append(episode_return)
            eval_costs.append(episode_cost)
            eval_discounted_returns.append(discounted_return)
            eval_discounted_costs.append(discounted_cost)
        
        # Compute stats for this initial stock
        stock_results = {
            'mean_return': float(np.mean(eval_returns)),
            'std_return': float(np.std(eval_returns)),
            'mean_cost': float(np.mean(eval_costs)),
            'std_cost': float(np.std(eval_costs)),
            'mean_discounted_return': float(np.mean(eval_discounted_returns)),
            'std_discounted_return': float(np.std(eval_discounted_returns)),
            'mean_discounted_cost': float(np.mean(eval_discounted_costs)),
            'std_discounted_cost': float(np.std(eval_discounted_costs)),
            'returns': [float(r) for r in eval_returns],
            'costs': [float(c) for c in eval_costs],
            'discounted_returns': [float(r) for r in eval_discounted_returns],
            'discounted_costs': [float(c) for c in eval_discounted_costs],
        }
        per_stock_results[float(initial_stock)] = stock_results
        
        # Aggregate all results
        all_returns.extend(eval_returns)
        all_costs.extend(eval_costs)
        all_discounted_returns.extend(eval_discounted_returns)
        all_discounted_costs.extend(eval_discounted_costs)
        
        print(f"  Return: {stock_results['mean_return']:.2f} ± {stock_results['std_return']:.2f}")
        print(f"  Cost: {stock_results['mean_cost']:.2f} ± {stock_results['std_cost']:.2f}")
    
    env.close()
    
    # Compute aggregate statistics
    aggregate = {
        'mean_return': float(np.mean(all_returns)),
        'std_return': float(np.std(all_returns)),
        'mean_cost': float(np.mean(all_costs)),
        'std_cost': float(np.std(all_costs)),
        'mean_discounted_return': float(np.mean(all_discounted_returns)),
        'std_discounted_return': float(np.std(all_discounted_returns)),
        'mean_discounted_cost': float(np.mean(all_discounted_costs)),
        'std_discounted_cost': float(np.std(all_discounted_costs)),
        'total_episodes': len(all_returns),
    }
    
    print(f"\n{'='*70}")
    print(f"Multi-Stock Evaluation Summary:")
    print(f"{'='*70}")
    print(f"{'Initial Stock':>15} | {'Return':>20} | {'Cost':>20}")
    print(f"{'-'*15}-+-{'-'*20}-+-{'-'*20}")
    for stock in sorted(per_stock_results.keys()):
        res = per_stock_results[stock]
        print(f"{stock:>15.1f} | {res['mean_return']:>8.2f} ± {res['std_return']:<8.2f} | {res['mean_cost']:>8.2f} ± {res['std_cost']:<8.2f}")
    print(f"{'-'*15}-+-{'-'*20}-+-{'-'*20}")
    print(f"{'Aggregate':>15} | {aggregate['mean_return']:>8.2f} ± {aggregate['std_return']:<8.2f} | {aggregate['mean_cost']:>8.2f} ± {aggregate['std_cost']:<8.2f}")
    print(f"{'='*70}\n")
    
    results = {
        'per_stock_results': per_stock_results,
        'aggregate': aggregate,
        'initial_cost_stocks': [float(s) for s in initial_cost_stocks],
        'gamma': gamma,
    }
    
    # Always save to cache (use_cache only controls whether to load)
    cache_data = {
        'eval_params': eval_params,
        'results': results
    }
    with open(cache_path, 'w') as f:
        json.dump(cache_data, f, indent=2)
    print(f"✓ Multi-stock evaluation results cached to: {cache_path}")
    
    return results

