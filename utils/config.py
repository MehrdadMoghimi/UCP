"""Config loader for environment-specific hyperparameter defaults."""

import inspect
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar

import yaml

# Type variable for generic dataclass support
T = TypeVar('T')


def _get_caller_script_name() -> Optional[str]:
    """
    Detect the name of the calling script (e.g., 'ucp' from 'ucp.py').
    
    Returns:
        Script name without .py extension, or None if not detectable
    """
    # First, check sys.argv[0] (most reliable when set correctly by tuner)
    if sys.argv and sys.argv[0].endswith('.py'):
        return Path(sys.argv[0]).stem
    
    # Fallback: Walk up the call stack to find the main script
    for frame_info in inspect.stack():
        frame_filename = frame_info.filename
        if frame_filename.endswith('.py') and '__main__' in (frame_info.code_context[0] if frame_info.code_context else ''):
            script_path = Path(frame_filename)
            return script_path.stem  # filename without extension
    
    return None


def load_env_config(env_id: str, config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load environment-specific defaults from YAML config.
    
    Automatically detects the config file from the calling script name
    (e.g., ``ucp.py`` -> ``configs/ucp.yaml``). If that file does not exist,
    dataclass defaults are used.
    
    Args:
        env_id: The environment ID (e.g., "SafetyCarGoal1-v0")
        config_path: Optional explicit path to config file. If None, auto-detects.
    
    Returns:
        Dictionary of hyperparameter defaults for the environment.
        Falls back to "defaults" key if env_id not found.
    
    Example:
        >>> config = load_env_config("SafetyCarGoal1-v0")
        >>> print(config.get("policy_lr", 3e-4))
    """
    # Resolve path relative to repo root
    repo_root = Path(__file__).parent.parent
    
    # Auto-detect config file if not provided
    if config_path is None:
        script_name = _get_caller_script_name()
        if script_name:
            config_path = f"configs/{script_name}.yaml"
            print(f"ℹ️  Auto-detected config file: {config_path}")
        else:
            # Fallback to generic config
            config_path = "configs/envs.yaml"
            print(f"⚠️  Could not auto-detect script. Using fallback: {config_path}")
    
    config_file = repo_root / config_path
    
    if not config_file.exists():
        print(f"⚠️  Config file not found: {config_file}. Using built-in defaults.")
        return {}
    
    try:
        with open(config_file, "r") as f:
            all_configs = yaml.safe_load(f)
    except Exception as e:
        print(f"⚠️  Error reading config file: {e}. Using built-in defaults.")
        return {}
    
    # Return environment-specific config, or fall back to "defaults"
    if env_id in all_configs:
        print(f"✓ Loaded config for '{env_id}' from {config_file.name}")
        return all_configs[env_id]
    elif "defaults" in all_configs:
        print(f"ℹ️  No config for '{env_id}', using defaults from {config_file.name}.")
        return all_configs["defaults"]
    else:
        print(f"⚠️  No 'defaults' found in config. Proceeding with Args defaults.")
        return {}


def parse_args_with_config(args_class: type[T]) -> T:
    """
    Parse CLI arguments and merge with environment-specific config from YAML.
    
    This function:
    1. Parses CLI arguments using tyro
    2. Loads environment-specific config based on --env-id
    3. Merges them with CLI taking precedence over YAML config
    4. Ensures proper type conversion from YAML values
    
    Priority order (highest to lowest):
    1. Explicit CLI arguments (anything in sys.argv)
    2. Environment-specific YAML config
    3. Dataclass defaults
    
    Args:
        args_class: The dataclass type (e.g., Args)
    
    Returns:
        Instance of args_class with merged configuration
        
    Example:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Args:
        ...     env_id: str = "CartPole-v1"
        ...     learning_rate: float = 1e-3
        >>> args = parse_args_with_config(Args)
        >>> # If configs/envs.yaml has learning_rate: 5e-4 for env_id,
        >>> # and user didn't pass --learning-rate, it will be 5e-4
    """
    import tyro
    from dataclasses import fields
    
    # Detect which arguments were explicitly passed via CLI by checking sys.argv
    # Convert field names to CLI argument format (snake_case -> --kebab-case)
    cli_provided_args = set()
    for arg in sys.argv[1:]:  # Skip script name
        if arg.startswith('--'):
            # Remove '--' prefix
            arg_name = arg[2:]
            # Handle --no- prefix for boolean flags
            if arg_name.startswith('no-'):
                arg_name = arg_name[3:]  # Remove 'no-' prefix
            # Convert to snake_case
            arg_name = arg_name.replace('-', '_')
            cli_provided_args.add(arg_name)
    
    # First pass: parse CLI to get all arguments
    temp_args = tyro.cli(args_class)
    
    # Load env-specific config based on parsed env_id
    env_config = load_env_config(temp_args.env_id)
    
    if not env_config:
        # No config found, just return parsed args
        return temp_args
    
    # Get field types for proper type conversion
    field_types = {f.name: f.type for f in fields(args_class)}
    
    # Priority: CLI args > YAML config > dataclass defaults
    merged_dict = {}
    default_args = args_class()  # Get default values from dataclass
    
    for field_name in vars(default_args):
        temp_value = getattr(temp_args, field_name)
        default_value = getattr(default_args, field_name)
        
        # Check if this argument was explicitly provided via CLI
        if field_name in cli_provided_args:
            # CLI argument takes highest priority - always use it
            merged_dict[field_name] = temp_value
        elif field_name in env_config:
            # YAML config value (second priority)
            config_value = env_config[field_name]
            # Ensure proper type conversion from YAML
            if field_name in field_types:
                expected_type = field_types[field_name]
                # Handle type conversion
                try:
                    if expected_type == bool:
                        merged_dict[field_name] = bool(config_value)
                    elif expected_type == int:
                        merged_dict[field_name] = int(config_value)
                    elif expected_type == float:
                        merged_dict[field_name] = float(config_value)
                    elif expected_type == str:
                        merged_dict[field_name] = str(config_value)
                    else:
                        merged_dict[field_name] = config_value
                except (ValueError, TypeError) as e:
                    print(f"⚠️  Warning: Could not convert {field_name}={config_value} to {expected_type}. Using default.")
                    merged_dict[field_name] = default_value
            else:
                merged_dict[field_name] = config_value
        else:
            # Dataclass default (lowest priority)
            merged_dict[field_name] = default_value
    
    # Create final args with merged values
    final_args = replace(temp_args, **merged_dict)
    
    return final_args


def merge_configs(cli_args: Dict[str, Any], env_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge CLI arguments with environment config, with CLI taking precedence.
    
    Args:
        cli_args: Arguments from tyro.cli() (may be incomplete dict or have sys.argv)
        env_config: Environment-specific defaults from YAML
    
    Returns:
        Merged configuration where CLI args override env_config
    
    Example:
        >>> env_config = {"learning_rate": 5e-4, "batch_size": 64}
        >>> cli_args = {"learning_rate": 1e-3}  # user override
        >>> merged = merge_configs(cli_args, env_config)
        >>> merged["learning_rate"]  # 1e-3 (CLI wins)
        >>> merged["batch_size"]  # 64 (from env_config)
    """
    # Start with env config as base
    merged = env_config.copy()
    
    # Override with any CLI args that were explicitly provided
    if cli_args:
        for key, value in cli_args.items():
            if value is not None:  # Only override if user provided a value
                merged[key] = value
    
    return merged


def apply_config_to_dataclass(dataclass_type: type, env_id: str, cli_args: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Convenience function: load env config and merge with CLI args.
    
    Args:
        dataclass_type: The Args dataclass type (for reference only, unused)
        env_id: The environment ID
        cli_args: CLI arguments from tyro.cli()
    
    Returns:
        Dictionary of all parameters ready to instantiate the dataclass
    """
    env_config = load_env_config(env_id)
    merged = merge_configs(cli_args or {}, env_config)
    return merged


__all__ = [
    "load_env_config",
    "parse_args_with_config",
    "merge_configs",
    "apply_config_to_dataclass",
]
