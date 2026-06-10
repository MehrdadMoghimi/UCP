"""Pretty-print utilities for training configurations."""

from dataclasses import asdict, is_dataclass
from typing import Any, Dict


def pretty_print_args(args: Any, title: str = "Training Configuration") -> None:
    """
    Pretty-print training arguments with nice formatting.
    
    Args:
        args: Args dataclass or dict to print
        title: Title for the output section
    
    Example:
        >>> from ucp import Args
        >>> args = Args(env_id="CartPole-v1", learning_rate=1e-3)
        >>> pretty_print_args(args)
    """
    # Convert dataclass to dict if needed
    if is_dataclass(args):
        args_dict = asdict(args)
    else:
        args_dict = args
    
    # Separate into categories for readability
    categories = {
        "Experiment": [
            "exp_name",
            "seed",
            "dir",
            "track",
            "wandb_project_name",
            "wandb_entity",
            "use_tb",
            "capture_video",
        ],
        "Environment": [
            "env_id",
            "num_envs",
            "total_timesteps",
        ],
        "Replay Buffer": [
            "buffer_size",
            "lagrange_buffer_size",
            "batch_size",
            "learning_starts",
        ],
        "N-Step & Discounting": [
            "n_step",
            "gamma",
        ],
        "Learning Rates": [
            "policy_lr",
            "q_lr",
            "c_lr",
            "alpha_lr",
            "lambda_lr",
        ],
        "Target Networks": [
            "q_tau",
            "c_tau",
            "p_tau",
            "target_network_frequency",
            "policy_frequency",
            "target_policy_update_freq",
        ],
        "Entropy & Policy": [
            "alpha",
            "autotune",
            "policy_reg_weight",
        ],
        "Safety & Lagrangian": [
            "lagrangian_multiplier_init",
            "lagrangian_upper_bound",
            "lagrangian_lower_bound",
            "lagrange_warmup_steps",
            "contextual_lambda_bins",
            "cost_utility_type",
            "lagrangian_utility_type",
            "lagrangian_utility_epsilon",
        ],
        "Observation Augmentation": [
            "add_cost_stock",
            "sample_initial_cost_stock",
            "eval_initial_cost_stocks",
        ],
        "Stock Parameters": [
            "gamma_cost",
            "initial_cost_stock",
            "cost_normalizer",
            "cost_stock_min",
            "cost_stock_max",
        ],
        "Network Architecture": [
            "n_quantiles",
            "actor_hidden_sizes",
            "critic_hidden_sizes",
            "embed_layers",
            "embed_connection_type",
            "embed_connection_layer",
            "use_embedding",
        ],
        "Model Saving & Evaluation": [
            "save_model",
            "save_freq",
            "evaluation_episodes",
            "evaluation_temperature",
        ],
        "Gradient Clipping": [
            "actor_max_grad_norm",
            "qf_max_grad_norm",
            "cost_qf_max_grad_norm",
        ],
        "PyTorch Performance": [
            "cuda",
            "torch_deterministic",
            "compile",
            "cudagraphs",
            "measure_burnin",
        ],
    }
    
    # Print header
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)
    
    # Track which keys we've printed
    printed_keys = set()
    
    # Print categorized args
    for category, keys in categories.items():
        # Filter to only keys that exist in args_dict
        relevant_keys = [k for k in keys if k in args_dict]
        
        if relevant_keys:
            print(f"\n  {category}:")
            print("  " + "-" * 76)
            
            for key in relevant_keys:
                value = args_dict[key]
                _print_arg(key, value, indent=4)
                printed_keys.add(key)
    
    # Print any uncategorized keys
    uncategorized = set(args_dict.keys()) - printed_keys
    if uncategorized:
        print(f"\n  Other:")
        print("  " + "-" * 76)
        for key in sorted(uncategorized):
            _print_arg(key, args_dict[key], indent=4)
    
    print("\n" + "=" * 80 + "\n")


def _print_arg(key: str, value: Any, indent: int = 0) -> None:
    """
    Print a single argument with type-aware formatting.
    
    Args:
        key: Argument name
        value: Argument value
        indent: Indentation level (in spaces)
    """
    indent_str = " " * indent
    
    # Format the value based on its type
    if isinstance(value, bool):
        formatted_value = "✓ Yes" if value else "✗ No"
        color = "\033[92m" if value else "\033[91m"  # Green if True, Red if False
        reset = "\033[0m"
        print(f"{indent_str}{key:<35} {color}{formatted_value}{reset}")
    elif isinstance(value, float):
        if value < 0.001 and value != 0:
            print(f"{indent_str}{key:<35} {value:.2e}")
        else:
            print(f"{indent_str}{key:<35} {value:.6g}")
    elif isinstance(value, int):
        print(f"{indent_str}{key:<35} {value:,}")  # Add thousands separator
    elif isinstance(value, str):
        print(f"{indent_str}{key:<35} '{value}'")
    else:
        print(f"{indent_str}{key:<35} {value}")


def print_config_comparison(cli_args: Dict[str, Any], env_config: Dict[str, Any], title: str = "Config Merge") -> None:
    """
    Print side-by-side comparison of CLI args vs environment config.
    
    Args:
        cli_args: Arguments provided via CLI
        env_config: Defaults loaded from YAML
        title: Title for the output
    
    Example:
        >>> from cleanrl_utils.config import load_env_config
        >>> env_config = load_env_config("AmericanOptionEnv-v1")
        >>> cli_args = {"learning_rate": 1e-2, "batch_size": 256}
        >>> print_config_comparison(cli_args, env_config)
    """
    print("\n" + "=" * 100)
    print(f"  {title}")
    print("=" * 100)
    
    all_keys = set(cli_args.keys()) | set(env_config.keys())
    
    print(f"\n  {'Parameter':<30} {'CLI / Env Config':<35} {'Final Value':<30}")
    print("  " + "-" * 96)
    
    for key in sorted(all_keys):
        cli_val = cli_args.get(key, "—")
        env_val = env_config.get(key, "—")
        final_val = cli_val if cli_val != "—" else env_val
        
        # Format the display
        cli_str = _format_value(cli_val)
        env_str = _format_value(env_val)
        final_str = _format_value(final_val)
        
        # Highlight where CLI overrides env
        if cli_val != "—" and env_val != "—" and cli_val != env_val:
            override_marker = " → "
            print(f"  {key:<30} {cli_str:>15} {override_marker} {env_str:<15} {final_str:>30}")
        else:
            print(f"  {key:<30} {cli_str:>33} {final_str:>30}")
    
    print("=" * 100 + "\n")


def _format_value(value: Any) -> str:
    """Format a value for display, handling None and special types."""
    if value == "—":
        return "—"
    elif isinstance(value, float):
        if value < 0.001 and value != 0:
            return f"{value:.2e}"
        else:
            return f"{value:.6g}"
    elif isinstance(value, bool):
        return "✓" if value else "✗"
    elif isinstance(value, int):
        return f"{value:,}"
    elif isinstance(value, str):
        return f"'{value}'"
    else:
        return str(value)


__all__ = [
    "pretty_print_args",
    "print_config_comparison",
]
