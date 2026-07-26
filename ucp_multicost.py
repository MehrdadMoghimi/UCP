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
#
# This file is based on `sac_continuous_action_torchcompile.py` from LeanRL
# (https://github.com/meta-pytorch/LeanRL), originally licensed under the MIT License.
#
# Multi-cost variant of `ucp.py`. Safety Gymnasium only returns a scalar cost, so
# `utils.multi_cost_wrappers.MultiCostWrapper` splits it into a vector of C cost
# dimensions (native per-obstacle terms and/or synthetic ones). Everything that was
# a single cost signal here becomes C-dimensional:
#
#   * one cost stock per dimension in the observation,
#   * an ensemble of C independent distributional cost critics (vmapped, exactly
#     like the existing double reward critic),
#   * one contextual multiplier λ_i(c₀_i) per dimension; the actor pays
#     Σ_i λ_i·U^c_i and is normalized by 1 + Σ_i λ_i.
#
# With a single cost dimension this file is equivalent to `ucp.py`.

import os

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

import math
import random
import time
from collections import deque
from dataclasses import dataclass
from functools import partial
from typing import Optional

import safety_gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import tyro
from tensordict import TensorDict, from_module, from_modules
from tensordict.nn import CudaGraphModule, TensorDictModule
from torch.utils.tensorboard import SummaryWriter

from utils.multi_cost_wrappers import (
    MultiCostAugmentedObservation,
    MultiCostWrapper,
    broadcast_to_costs,
)
from utils.config import parse_args_with_config
from utils.pretty_print import pretty_print_args
from utils.evaluator_multicost import evaluate_ucp_multicost, evaluate_ucp_multicost_multi_stock

from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.envs.transforms import MultiStepTransform
torch.backends.cudnn.conv.fp32_precision = 'tf32'

import warnings 
warnings.filterwarnings("ignore", category=UserWarning)

@dataclass
class Args:
    exp_name: str = "UCP-MC"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "UCP-MC"
    """the wandb's project name"""
    wandb_entity: str = "UCP"
    """the entity (team) of wandb's project"""
    use_tb: bool = True
    """if toggled, TensorBoard logging will be enabled"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    num_envs: int = 8
    """number of parallel safety environments"""
    dir: str = "runs"
    """the directory where the experiment data will be stored"""

    # Algorithm specific arguments
    env_id: str = "SafetyCarGoal1-v0"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    lagrange_buffer_size: int = 8
    """the replay memory buffer size for Lagrange multiplier"""
    gamma: float = 0.99
    """the discount factor gamma"""
    q_tau: float = 0.005
    """target smoothing coefficient for reward Q"""
    c_tau: float = 0.005
    """target smoothing coefficient for cost Q"""
    p_tau: float = 0.005
    """target smoothing coefficient for policy"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 30000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-4
    """the learning rate of the Q network network optimizer"""
    c_lr: float = 1e-4
    """the learning rate of the cost Q network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1 
    """the frequency of updates for the target nerworks"""
    lagrangian_multiplier_init: float = 1.0
    """initial value for the Lagrange multiplier"""
    lambda_lr: float = 5e-5
    """learning rate for the Lagrange multiplier"""
    lagrangian_upper_bound: Optional[float] = None
    """optional upper bound for the Lagrange multiplier"""
    lagrangian_lower_bound: float = 1e-3
    """minimum threshold for the Lagrange multiplier"""
    lagrange_warmup_steps: int = 50000
    """gradient steps to wait before updating the Lagrange multiplier"""
    
    # Contextual Lagrangian arguments (enabled when sample_initial_cost_stock=True)
    contextual_lambda_bins: int = 31
    """number of bins for discretizing initial cost stock range [cost_stock_min, cost_stock_max] for contextual λ"""
    n_quantiles: int = 100
    """the number of quantiles for distributional critic"""
    
    # N-step learning arguments
    n_step: int = 1
    """the number of steps to look ahead for n-step Q learning"""
    
    # Policy regularization arguments
    policy_reg_weight: float = 1e-3
    """Regularization weight for policy mean and std"""
    target_policy_update_freq: int = 1
    """Frequency (in gradient steps) to update the target actor"""
    
    # Entropy arguments (SAC-style)
    autotune: bool = False
    """whether to autotune the entropy coefficient alpha"""
    alpha: float = 0.0
    """entropy regularization coefficient (if autotune=False, this is the fixed value; if autotune=True, this is the initial value)"""
    alpha_lr: float = 5e-5
    """the learning rate of the entropy coefficient alpha"""
    
    # Multi-cost arguments
    cost_specs: list[str] = None
    """cost dimensions exposed by MultiCostWrapper (default: ["auto"] = native decomposition).
    Examples: ["auto"], ["native:hazards", "native:gremlins"], ["sum", "ctrl:0.5"], ["sum", "speed:0.3"]"""
    cost_normalizers: list[float] = None
    """per-dimension override for cost_normalizer (None broadcasts the scalar to every dimension)"""
    initial_cost_stocks: list[float] = None
    """per-dimension override for initial_cost_stock (None broadcasts the scalar)"""
    cost_stock_mins: list[float] = None
    """per-dimension override for cost_stock_min (None broadcasts the scalar)"""
    cost_stock_maxs: list[float] = None
    """per-dimension override for cost_stock_max (None broadcasts the scalar)"""

    # Observation augmentation wrapper arguments
    add_cost_stock: bool = True
    """if toggled, one cost stock per cost dimension will be added to observations"""
    gamma_cost: float = 1.0
    """Discount factor for cost stock accumulation"""
    initial_cost_stock: float = 0.0
    """Initial value for cost stock (used when sample_initial_cost_stock=False)"""
    cost_utility_type: str = "abs"
    """the type of cost utility function: 'abs', 'mean'"""
    lagrangian_utility_type: str = "abs"
    """the type of lagrangian utility function: 'abs' or 'mean'"""
    lagrangian_utility_epsilon: float = 1e-6
    """small epsilon for numerical stability in lagrangian utility calculation when using 'abs' type"""
    sample_initial_cost_stock: bool = True
    """if toggled, sample initial cost stock uniformly from [cost_stock_min, cost_stock_max] at each episode reset during training"""
    eval_initial_cost_stocks: list[float] = None
    """list of initial cost stock values to evaluate with (e.g., [-25, -20, -15, -10, -5, 0]).
    Each value is applied to every cost dimension; if None, uses initial_cost_stock"""
    cost_normalizer: float = 10.0
    """Normalization factor for cost stock in observations"""
    cost_stock_min: float = -30.0
    """the minimum stock for searching optimal stock"""
    cost_stock_max: float = 0.0
    """the maximum stock for searching optimal stock"""

    compile: bool = True
    """whether to use torch.compile."""
    cudagraphs: bool = True
    """whether to use cudagraphs on top of compile."""

    measure_burnin: int = 3
    """Number of burn-in iterations for speed measure."""
    
    # Network architecture arguments
    actor_hidden_sizes: list[int] = None
    """hidden layer sizes for actor network (default: [256, 256])"""
    critic_hidden_sizes: list[int] = None
    """hidden layer sizes for critic networks (default: [256, 256])"""
    
    # Embedding arguments
    embed_layers: list[int] = None
    """hidden layer sizes for embedding network (default: [16])"""
    embed_connection_type: str = "concat"
    """how to connect embedding to main network: 'concat' or 'add'"""
    embed_connection_layer: int = 0
    """which layer (0-indexed) of main network to connect embedding to"""
    use_embedding: bool = True
    """whether to use embeddings for augmented features"""
    
    # Model saving and evaluation arguments
    save_model: bool = True
    """whether to save the model at the end of training"""
    save_freq: int = 1000000
    """save model checkpoint every K timesteps (only if save_model=True)"""
    evaluation_episodes: int = 1000
    """number of episodes to evaluate the model (0 means no evaluation)"""
    evaluation_temperature: float = 0.0
    """temperature parameter for policy action sampling (1.0=normal, 0.0=deterministic)"""
    actor_max_grad_norm: Optional[float] = None
    """maximum gradient norm for actor clipping (None disables clipping)"""
    qf_max_grad_norm: Optional[float] = 0.05
    """maximum gradient norm for reward critic clipping (None disables clipping)"""
    cost_qf_max_grad_norm: Optional[float] = 0.05
    """maximum gradient norm for cost critic clipping (None disables clipping)"""
    
    def __post_init__(self):
        """Set default values for mutable arguments."""
        if self.actor_hidden_sizes is None:
            self.actor_hidden_sizes = [256, 256]
        if self.critic_hidden_sizes is None:
            self.critic_hidden_sizes = [256, 256]
        if self.embed_layers is None:
            self.embed_layers = [16]
        if self.eval_initial_cost_stocks is None:
            self.eval_initial_cost_stocks = [-25, -15, -5, 0]
        if self.cost_specs is None:
            self.cost_specs = ["auto"]


class SoftQNetwork(nn.Module):
    def __init__(self, env, n_act, n_obs, n_quantiles, device=None, hidden_sizes=None,
                 use_embedding=False, embed_layers=None, embed_connection_type="concat", 
                 embed_connection_layer=0, n_base_obs=None, n_aug_features=0):
        super().__init__()
        self.use_embedding = use_embedding
        self.n_base_obs = n_base_obs if n_base_obs is not None else n_obs
        self.n_aug_features = n_aug_features
        self.n_act = n_act
        self.embed_connection_type = embed_connection_type
        self.embed_connection_layer = embed_connection_layer
        
        if hidden_sizes is None:
            hidden_sizes = [256, 256]
        
        if embed_layers is None:
            embed_layers = [16]
        
        # Create embedding network if augmented features are used
        self.embed_net = None
        self.embed_output_dim = 0
        if use_embedding and n_aug_features > 0:
            embed_net_layers = nn.ModuleList()
            prev_size = n_aug_features
            for embed_size in embed_layers:
                embed_net_layers.append(nn.Linear(prev_size, embed_size, device=device))
                prev_size = embed_size
            self.embed_net = embed_net_layers
            self.embed_output_dim = embed_layers[-1]  # Last layer dimension
            
            # Validate connection layer
            if embed_connection_layer < 0 or embed_connection_layer >= len(hidden_sizes):
                raise ValueError(f"embed_connection_layer must be in [0, {len(hidden_sizes)-1}]")
            
            # For 'add' mode, check dimension compatibility
            if embed_connection_type == "add":
                if self.embed_output_dim != hidden_sizes[embed_connection_layer]:
                    raise ValueError(
                        f"For 'add' connection, embedding output dim ({self.embed_output_dim}) "
                        f"must match hidden layer {embed_connection_layer} dim ({hidden_sizes[embed_connection_layer]})"
                    )
        
        # Build main network layers
        self.layers = nn.ModuleList()
        
        # First layer: obs + action
        # When embedding is disabled, use full observation; otherwise use base observations
        obs_dim_for_network = n_obs if not use_embedding else self.n_base_obs
        feature_input_dim = n_act + obs_dim_for_network
        prev_size = feature_input_dim
        
        for i, hidden_size in enumerate(hidden_sizes):
            # Add embedding contribution at the specified layer
            if i == embed_connection_layer and self.use_embedding and self.embed_net is not None:
                if embed_connection_type == "concat":
                    # Concatenate: increase input size
                    self.layers.append(nn.Linear(prev_size + self.embed_output_dim, hidden_size, device=device))
                else:  # "add"
                    # Add: no change to dimensions, will add after linear layer
                    self.layers.append(nn.Linear(prev_size, hidden_size, device=device))
            else:
                self.layers.append(nn.Linear(prev_size, hidden_size, device=device))
            prev_size = hidden_size
        
        # Output layer
        self.fc_out = nn.Linear(prev_size, n_quantiles, device=device)
        
        self.n_quantiles = n_quantiles

    def forward(self, x, a):
        # Process embedding if enabled
        embed_output = None
        if self.use_embedding and self.embed_net is not None and self.n_aug_features > 0:
            # Split: base observations and augmented features
            base_obs = x[:, :self.n_base_obs]
            aug_features = x[:, self.n_base_obs:]
            
            # Process through embedding network
            embed_output = aug_features
            for embed_layer in self.embed_net:
                embed_output = F.relu(embed_layer(embed_output))
            
            # Use only base observations for main network input
            x = base_obs
        # If embedding is disabled, use the full observation (no splitting)
        
        # Concatenate obs with action
        x = torch.cat([x, a], 1)
        
        # Process through main network layers
        for i, layer in enumerate(self.layers):
            if i == self.embed_connection_layer and embed_output is not None:
                # Before applying activation, connect embedding
                if self.embed_connection_type == "concat":
                    # Concatenate embedding before linear layer
                    x = torch.cat([x, embed_output], dim=1)
                    x = F.relu(layer(x))
                else:  # "add"
                    # Apply linear layer then add embedding
                    x = F.relu(layer(x) + embed_output)
            else:
                x = F.relu(layer(x))
        
        x = self.fc_out(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5
EPS = 1e-6

class Actor(nn.Module):
    def __init__(self, action_space, n_obs, n_act, device=None, hidden_sizes=None, 
                 use_embedding=False, embed_layers=None, embed_connection_type="concat",
                 embed_connection_layer=0, n_base_obs=None, n_aug_features=0):
        super().__init__()
        self.use_embedding = use_embedding
        self.n_base_obs = n_base_obs if n_base_obs is not None else n_obs
        self.n_aug_features = n_aug_features
        self.embed_connection_type = embed_connection_type
        self.embed_connection_layer = embed_connection_layer
        
        if hidden_sizes is None:
            hidden_sizes = [256, 256]
        
        if embed_layers is None:
            embed_layers = [16]
        
        # Create embedding network if augmented features are used
        self.embed_net = None
        self.embed_output_dim = 0
        if use_embedding and n_aug_features > 0:
            embed_net_layers = nn.ModuleList()
            prev_size = n_aug_features
            for embed_size in embed_layers:
                embed_net_layers.append(nn.Linear(prev_size, embed_size, device=device))
                prev_size = embed_size
            self.embed_net = embed_net_layers
            self.embed_output_dim = embed_layers[-1]  # Last layer dimension
            
            # Validate connection layer
            if embed_connection_layer < 0 or embed_connection_layer >= len(hidden_sizes):
                raise ValueError(f"embed_connection_layer must be in [0, {len(hidden_sizes)-1}]")
            
            # For 'add' mode, check dimension compatibility
            if embed_connection_type == "add":
                if self.embed_output_dim != hidden_sizes[embed_connection_layer]:
                    raise ValueError(
                        f"For 'add' connection, embedding output dim ({self.embed_output_dim}) "
                        f"must match hidden layer {embed_connection_layer} dim ({hidden_sizes[embed_connection_layer]})"
                    )
        
        # Build main network layers
        self.layers = nn.ModuleList()
        # When embedding is disabled, use full observation; otherwise use base observations
        prev_size = n_obs if not use_embedding else self.n_base_obs
        
        for i, hidden_size in enumerate(hidden_sizes):
            # Add embedding contribution at the specified layer
            if i == embed_connection_layer and self.use_embedding and self.embed_net is not None:
                if embed_connection_type == "concat":
                    # Concatenate: increase input size
                    self.layers.append(nn.Linear(prev_size + self.embed_output_dim, hidden_size, device=device))
                else:  # "add"
                    # Add: no change to dimensions, will add after linear layer
                    self.layers.append(nn.Linear(prev_size, hidden_size, device=device))
            else:
                self.layers.append(nn.Linear(prev_size, hidden_size, device=device))
            prev_size = hidden_size
        
        # Output layers for mean and log_std
        self.fc_mean = nn.Linear(prev_size, n_act, device=device)
        self.fc_logstd = nn.Linear(prev_size, n_act, device=device)
        
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32, device=device),
        )

    def forward(self, x):
        # Process embedding if enabled
        embed_output = None
        if self.use_embedding and self.embed_net is not None and self.n_aug_features > 0:
            # Split: base observations and augmented features
            base_obs = x[:, :self.n_base_obs]
            aug_features = x[:, self.n_base_obs:]
            
            # Process through embedding network
            embed_output = aug_features
            for embed_layer in self.embed_net:
                embed_output = F.relu(embed_layer(embed_output))
            
            # Use only base observations for main network input
            x = base_obs
        # If embedding is disabled, use the full observation (no splitting)
        
        # Process through main network layers
        for i, layer in enumerate(self.layers):
            if i == self.embed_connection_layer and embed_output is not None:
                # Before applying activation, connect embedding
                if self.embed_connection_type == "concat":
                    # Concatenate embedding before linear layer
                    x = torch.cat([x, embed_output], dim=1)
                    x = F.relu(layer(x))
                else:  # "add"
                    # Apply linear layer then add embedding
                    x = F.relu(layer(x) + embed_output)
            else:
                x = F.relu(layer(x))
        
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x, action=None, action_unsquashed=None, temperature=1.0):
        """
        Get action with standard Gaussian log probability.
        
        Args:
            x: observation
            action: if provided, compute log_prob for this action
            action_unsquashed: unsquashed version of action (before tanh)
            temperature: controls action stochasticity (1.0=normal, 0.0=deterministic mean)
        """
        mean, log_std = self(x)
        std = log_std.exp()
        
        if action is None:
            # Sample new action
            if temperature == 0.0:
                # Deterministic: use mean directly
                x_t = mean
            else:
                # Stochastic: sample with temperature-scaled std
                normal = torch.distributions.Normal(mean, std * temperature)
                x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
            action_unsquashed = x_t
        else:
            # Use provided action when evaluating its log probability.
            y_t = (action - self.action_bias) / self.action_scale
            if action_unsquashed is None:
                # Compute unsquashed action using inverse tanh (atanh)
                # Clamp y_t to avoid numerical issues with atanh at boundaries
                y_t_clamped = torch.clamp(y_t, -0.9999, 0.9999)
                x_t = torch.atanh(y_t_clamped)
            else:
                x_t = action_unsquashed
        
        # Standard log probability (SAC style)
        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        mean_squashed = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_squashed, action_unsquashed, mean, log_std


class ContextualLagrangian(nn.Module):
    """
    Binned contextual Lagrangian multipliers λ_i(c₀_i), one per cost dimension.

    When initial cost stocks are sampled, each episode starts with a budget vector c₀.
    This module discretizes each dimension's configured c₀ range into bins and learns
    one multiplier per (dimension, bin), so tighter budgets can receive stronger safety
    pressure and each constraint gets its own price.

    The multipliers are independent across dimensions: dimension i is priced only by
    its own budget c₀_i. Coupling happens in the actor loss, which pays Σ_i λ_i·U^c_i.

    Args:
        c0_min: Minimum initial cost stock value (scalar or one value per dimension)
        c0_max: Maximum initial cost stock value (scalar or one value per dimension)
        num_costs: Number of cost dimensions
        num_bins: Number of bins for discretization
        init_value: Initial value for all multipliers
        lower_bound: Minimum allowed value for λ
        upper_bound: Maximum allowed value for λ (None for unbounded)
        device: Torch device
    """

    def __init__(
        self,
        c0_min,
        c0_max,
        num_costs: int = 1,
        num_bins: int = 11,
        init_value: float = 1.0,
        lower_bound: float = 1e-3,
        upper_bound: float = None,
        device: torch.device = None,
    ):
        super().__init__()
        self.num_costs = num_costs
        self.num_bins = num_bins
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.device = device

        # Per-dimension c₀ ranges, kept as buffers so they follow the module's device.
        self.register_buffer(
            "c0_min",
            torch.as_tensor(broadcast_to_costs(c0_min, num_costs, "c0_min"), device=device),
        )
        self.register_buffer(
            "c0_max",
            torch.as_tensor(broadcast_to_costs(c0_max, num_costs, "c0_max"), device=device),
        )

        # Store unconstrained parameters and map through softplus for positive λ.
        init_log = math.log(math.exp(init_value) - 1) if init_value > 0 else 0.0
        self.log_lambdas = nn.Parameter(
            torch.full((num_costs, num_bins), init_log, device=device, dtype=torch.float32)
        )

    def _c0_to_bin_idx(self, c0: torch.Tensor) -> torch.Tensor:
        """Convert c0 values, shape (batch_size, num_costs), to per-dimension bin indices."""
        # Normalize c0 to [0, 1] range using each dimension's own range
        c0_normalized = (c0 - self.c0_min) / (self.c0_max - self.c0_min + 1e-8)
        # Convert to bin index
        bin_idx = (c0_normalized * self.num_bins).long().clamp(0, self.num_bins - 1)
        return bin_idx

    def forward(self, c0: torch.Tensor) -> torch.Tensor:
        """
        Compute λ(c₀) for given initial cost stocks.

        Args:
            c0: Initial cost stock values, shape (batch_size, num_costs)
                (a 1D tensor is treated as a single-dimension batch)

        Returns:
            Lambda values, shape (batch_size, num_costs)
        """
        if c0.dim() == 1:
            c0 = c0.unsqueeze(-1)

        bin_idx = self._c0_to_bin_idx(c0)  # (batch_size, num_costs)
        # gather picks bin_idx[b, i] out of dimension i's row; gradients flow only to
        # the selected (dimension, bin) entries.
        log_lambda = self.log_lambdas.gather(1, bin_idx.t()).t()  # (batch_size, num_costs)
        lambda_val = F.softplus(log_lambda)

        # Apply bounds
        lambda_val = lambda_val.clamp(min=self.lower_bound)
        if self.upper_bound is not None:
            lambda_val = lambda_val.clamp(max=self.upper_bound)

        return lambda_val

    def get_all_lambdas(self) -> torch.Tensor:
        """Return the learned lambda value for every (dimension, bin), shape (num_costs, num_bins)."""
        lambda_vals = F.softplus(self.log_lambdas).clamp(min=self.lower_bound)
        if self.upper_bound is not None:
            lambda_vals = lambda_vals.clamp(max=self.upper_bound)
        return lambda_vals

    def get_bin_centers(self) -> torch.Tensor:
        """Return bin center values for logging, shape (num_costs, num_bins)."""
        return torch.stack([
            torch.linspace(self.c0_min[i].item(), self.c0_max[i].item(), self.num_bins, device=self.device)
            for i in range(self.num_costs)
        ])

if __name__ == "__main__":
    args = parse_args_with_config(Args)
    # Pretty-print args before training starts
    pretty_print_args(args, title=f"Training Configuration ({args.env_id})")
    
    run_name = (
        f"{args.env_id}__{args.exp_name}"
        f"__lag{args.lagrangian_multiplier_init:g}"
        f"__nstep{args.n_step}"
        f"__gamma{args.gamma:g}"
    )
        
    run_name += "__costs" + "-".join(str(s).replace(":", "") for s in args.cost_specs)
    if args.add_cost_stock:
        run_name += f"_cs"
    if args.autotune:
        run_name += f"__alpha{args.alpha:g}"
    if args.actor_max_grad_norm is not None or args.qf_max_grad_norm is not None or args.cost_qf_max_grad_norm is not None:
        run_name += f"__a{args.actor_max_grad_norm}_q{args.qf_max_grad_norm}_c{args.cost_qf_max_grad_norm}"
    if args.sample_initial_cost_stock:
        run_name += f"_binlag"
    run_name += f"__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    
    writer = None
    if args.use_tb:
        writer = SummaryWriter(f"{args.dir}/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    render_mode = "rgb_array" if args.capture_video else None

    # MultiCostWrapper turns the scalar cost into a vector; MultiCostAugmentedObservation
    # then appends one stock per cost dimension. Per-dimension settings fall back to the
    # scalar arguments when the *_s overrides are not given.
    multi_cost_wrapper = partial(
        MultiCostWrapper,
        cost_specs=args.cost_specs,
    )
    augmented_wrapper = partial(
        MultiCostAugmentedObservation,
        add_cost_stock=args.add_cost_stock,
        gamma_cost=args.gamma_cost,
        initial_cost_stock=args.initial_cost_stocks if args.initial_cost_stocks is not None else args.initial_cost_stock,
        cost_normalizer=args.cost_normalizers if args.cost_normalizers is not None else args.cost_normalizer,
        sample_initial_cost_stock=args.sample_initial_cost_stock,
        cost_stock_min=args.cost_stock_mins if args.cost_stock_mins is not None else args.cost_stock_min,
        cost_stock_max=args.cost_stock_maxs if args.cost_stock_maxs is not None else args.cost_stock_max,
        contextual_lambda_bins=args.contextual_lambda_bins,
    )
    action_wrapper = partial(gym.wrappers.SafeRescaleAction, min_action=-1.0, max_action=1.0)
    # asynchronous=True is required: the sync vector env preallocates a scalar cost buffer.
    envs = gym.vector.make(args.env_id, num_envs=args.num_envs, render_mode=render_mode, wrappers=[action_wrapper, multi_cost_wrapper, augmented_wrapper], asynchronous=True)

    cost_stock_index = envs.get_attr('cost_stock_index')[0]
    num_costs = envs.get_attr('num_costs')[0]
    cost_names = envs.get_attr('cost_names')[0]

    # Resolve every per-dimension setting now that the env told us how many costs it has.
    cost_normalizers_np = broadcast_to_costs(
        args.cost_normalizers if args.cost_normalizers is not None else args.cost_normalizer,
        num_costs, "cost_normalizer")
    initial_cost_stocks_np = broadcast_to_costs(
        args.initial_cost_stocks if args.initial_cost_stocks is not None else args.initial_cost_stock,
        num_costs, "initial_cost_stock")
    cost_stock_mins_np = broadcast_to_costs(
        args.cost_stock_mins if args.cost_stock_mins is not None else args.cost_stock_min,
        num_costs, "cost_stock_min")
    cost_stock_maxs_np = broadcast_to_costs(
        args.cost_stock_maxs if args.cost_stock_maxs is not None else args.cost_stock_max,
        num_costs, "cost_stock_max")
    cost_normalizers_t = torch.as_tensor(cost_normalizers_np, device=device)

    print(f"Multi-cost configuration:")
    print(f"  Cost dimensions ({num_costs}): {cost_names}")
    print(f"  Cost normalizers: {cost_normalizers_np.tolist()}")
    print(f"Observation augmentation configuration:")
    print(f"  Cost stock: {'enabled' if cost_stock_index is not None else 'disabled'} (start index: {cost_stock_index})")
    if args.sample_initial_cost_stock:
        print(f"  Sample initial cost stock: enabled (ranges: "
              f"{[(float(lo), float(hi)) for lo, hi in zip(cost_stock_mins_np, cost_stock_maxs_np)]})")
    else:
        print(f"  Initial cost stocks: {initial_cost_stocks_np.tolist()}")
    envs.single_action_space.seed(args.seed)
    n_act = math.prod(envs.single_action_space.shape)
    n_obs = math.prod(envs.single_observation_space.shape)

    n_aug_features = 0
    if args.add_cost_stock and cost_stock_index is not None:
        n_aug_features += num_costs

    # Calculate base observation dimension (original observations without augmentation)
    n_base_obs = n_obs - n_aug_features
    
    print(f"Network configuration:")
    print(f"  Total observation dim: {n_obs}")
    print(f"  Base observation dim: {n_base_obs}")
    print(f"  Augmented features: {n_aug_features}")
    print(f"  Use embedding: {args.use_embedding}")
    if args.use_embedding and n_aug_features > 0:
        print(f"  Embedding layers: {args.embed_layers}")
        print(f"  Embedding connection: {args.embed_connection_type} at layer {args.embed_connection_layer}")

    actor = Actor(envs.single_action_space, device=device, n_act=n_act, n_obs=n_obs, 
                  hidden_sizes=args.actor_hidden_sizes,
                  use_embedding=args.use_embedding, embed_layers=args.embed_layers,
                  embed_connection_type=args.embed_connection_type,
                  embed_connection_layer=args.embed_connection_layer,
                  n_base_obs=n_base_obs, n_aug_features=n_aug_features)
    actor_detach = Actor(envs.single_action_space, device=device, n_act=n_act, n_obs=n_obs,
                         hidden_sizes=args.actor_hidden_sizes,
                         use_embedding=args.use_embedding, embed_layers=args.embed_layers,
                         embed_connection_type=args.embed_connection_type,
                         embed_connection_layer=args.embed_connection_layer,
                         n_base_obs=n_base_obs, n_aug_features=n_aug_features)
    # Copy params to actor_detach without grad
    from_module(actor).data.to_module(actor_detach)
    
    # Create target actor
    def get_params_actor(actor):
        target_actor = Actor(envs.single_action_space, device="meta", n_act=n_act, n_obs=n_obs,
                             hidden_sizes=args.actor_hidden_sizes,
                             use_embedding=args.use_embedding, embed_layers=args.embed_layers,
                             embed_connection_type=args.embed_connection_type,
                             embed_connection_layer=args.embed_connection_layer,
                             n_base_obs=n_base_obs, n_aug_features=n_aug_features)
        actor_params = from_module(actor).data
        target_actor_params = actor_params.clone()
        target_actor_params.to_module(target_actor)
        return actor_params, target_actor_params, target_actor
    
    actor_params, target_actor_params, target_actor = get_params_actor(actor)
    
    # Wrapper for policy action selection (returns only action, not full tuple)
    def policy_forward(obs):
        action, _, _, _, _, _ = actor_detach.get_action(obs)
        return action
    
    policy = TensorDictModule(policy_forward, in_keys=["observation"], out_keys=["action"])

    def get_q_params():
        # Double distributional Q-network (like SAC) - use minimum for target
        qf1 = SoftQNetwork(envs, device=device, n_act=n_act, n_obs=n_obs, n_quantiles=args.n_quantiles,
                         hidden_sizes=args.critic_hidden_sizes,
                         use_embedding=args.use_embedding, embed_layers=args.embed_layers,
                         embed_connection_type=args.embed_connection_type,
                         embed_connection_layer=args.embed_connection_layer,
                         n_base_obs=n_base_obs, n_aug_features=n_aug_features)
        qf2 = SoftQNetwork(envs, device=device, n_act=n_act, n_obs=n_obs, n_quantiles=args.n_quantiles,
                         hidden_sizes=args.critic_hidden_sizes,
                         use_embedding=args.use_embedding, embed_layers=args.embed_layers,
                         embed_connection_type=args.embed_connection_type,
                         embed_connection_layer=args.embed_connection_layer,
                         n_base_obs=n_base_obs, n_aug_features=n_aug_features)
        qnet_params = from_modules(qf1, qf2, as_module=True)
        qnet_target = qnet_params.data.clone()

        # discard params of net
        qnet = SoftQNetwork(envs, device="meta", n_act=n_act, n_obs=n_obs, n_quantiles=args.n_quantiles,
                           hidden_sizes=args.critic_hidden_sizes,
                           use_embedding=args.use_embedding, embed_layers=args.embed_layers,
                           embed_connection_type=args.embed_connection_type,
                           embed_connection_layer=args.embed_connection_layer,
                           n_base_obs=n_base_obs, n_aug_features=n_aug_features)
        qnet_params.to_module(qnet)

        return qnet_params, qnet_target, qnet

    qnet_params, qnet_target, qnet = get_q_params()
    
    q_optimizer = optim.Adam(qnet.parameters(), lr=args.q_lr, capturable=args.cudagraphs and not args.compile)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr, capturable=args.cudagraphs and not args.compile)
    
    # Quantile midpoints for QR-DQN loss
    tau_hat = (2 * torch.arange(args.n_quantiles, device=device) + 1) / (2.0 * args.n_quantiles)
    tau_hat_reshaped = tau_hat.view(1, -1, 1).repeat(args.batch_size, 1, args.n_quantiles)
    def get_cost_q_params():
        # 1. One independent distributional cost critic per cost dimension. Stacking them
        #    with from_modules lets a single vmap evaluate all of them at once, exactly
        #    like the (qf1, qf2) reward pair above.
        cost_qfs = [
            SoftQNetwork(envs, device=device, n_act=n_act, n_obs=n_obs, n_quantiles=args.n_quantiles,
                         hidden_sizes=args.critic_hidden_sizes,
                         use_embedding=args.use_embedding, embed_layers=args.embed_layers,
                         embed_connection_type=args.embed_connection_type,
                         embed_connection_layer=args.embed_connection_layer,
                         n_base_obs=n_base_obs, n_aug_features=n_aug_features)
            for _ in range(num_costs)
        ]
        # 2. Extract their parameters into a functional container with a leading cost dim
        cost_qnet_params = from_modules(*cost_qfs, as_module=True)
        # 3. Clone them for the target network
        cost_qnet_target = cost_qnet_params.data.clone()
        # 4. Create the "hollow" meta network for the forward pass helper and optimizer
        cost_qnet = SoftQNetwork(envs, device="meta", n_act=n_act, n_obs=n_obs, n_quantiles=args.n_quantiles,
                                hidden_sizes=args.critic_hidden_sizes,
                                use_embedding=args.use_embedding, embed_layers=args.embed_layers,
                                embed_connection_type=args.embed_connection_type,
                                embed_connection_layer=args.embed_connection_layer,
                                n_base_obs=n_base_obs, n_aug_features=n_aug_features)
        cost_qnet_params.to_module(cost_qnet)
        return cost_qnet_params, cost_qnet_target, cost_qnet

    cost_qnet_params, cost_qnet_target, cost_qnet = get_cost_q_params()
    cost_optimizer = optim.Adam(cost_qnet.parameters(), lr=args.c_lr, capturable=args.cudagraphs and not args.compile)
    
    
    # Use a binned contextual multiplier when episodes start from sampled cost budgets.
    # Otherwise train one scalar multiplier shared by all samples.
    if args.sample_initial_cost_stock:
        contextual_lagrangian = ContextualLagrangian(
            c0_min=cost_stock_mins_np,
            c0_max=cost_stock_maxs_np,
            num_costs=num_costs,
            num_bins=args.contextual_lambda_bins,
            init_value=args.lagrangian_multiplier_init,
            lower_bound=args.lagrangian_lower_bound,
            upper_bound=args.lagrangian_upper_bound,
            device=device,
        )
        lambda_optimizer = optim.Adam(contextual_lagrangian.parameters(), lr=args.lambda_lr, capturable=args.cudagraphs and not args.compile)
        lagrangian_multiplier = None
    else:
        contextual_lagrangian = None
        # One multiplier per cost dimension.
        lagrangian_multiplier = torch.nn.Parameter(
            torch.full((num_costs,), max(float(args.lagrangian_multiplier_init), 0.0), device=device),
            requires_grad=True,
        )
        lambda_optimizer = optim.Adam([lagrangian_multiplier], lr=args.lambda_lr, capturable=args.cudagraphs and not args.compile)

    # Entropy tuning
    target_entropy = -torch.prod(torch.tensor(envs.single_action_space.shape, device=device)).item()
    log_alpha = torch.nn.Parameter(torch.tensor(math.log(max(args.alpha, 1e-8)), device=device), requires_grad=args.autotune)
    alpha_optimizer = optim.Adam([log_alpha], lr=args.alpha_lr, capturable=args.cudagraphs and not args.compile) if args.autotune else None

    envs.single_observation_space.dtype = np.float32
    
    # Initialize replay buffer with n-step transform
    # MultiStepTransform handles n-step returns automatically during extend()
    rb = TensorDictReplayBuffer(
        storage=LazyTensorStorage(args.buffer_size, device=device),
        batch_size=args.batch_size,
        # MultiStepTransform computes n-step returns during extend()
        transform=MultiStepTransform(
            n_steps=args.n_step,
            gamma=args.gamma,
            reward_keys=["rewards", "costs"],  # Apply n-step to both rewards and costs
            done_key="done",  # Must match the key we store in transitions
        ),
    )

    def batched_qf(params, obs, action, target_quantiles=None):
        """
        Returns quantiles from a single Q-network.
        When used with vmap over params from from_modules(qf1, qf2), returns shape (2, batch_size, n_quantiles).
        If target_quantiles provided, computes and returns the quantile Huber loss instead.
        """
        with params.to_module(qnet):
            quantiles = qnet(obs, action)
            if target_quantiles is not None:
                loss_val, _ = compute_quantile_regression_loss(quantiles, target_quantiles)
                return quantiles, loss_val
            return quantiles, quantiles.new_zeros(())
        
    def batched_cost_qf(params, obs, action, target_quantiles=None):
        """
        Returns quantiles from a single cost Q-network.
        When used with vmap over params from from_modules(*cost_qfs), returns shape
        (num_costs, batch_size, n_quantiles).
        If target_quantiles provided, computes and returns the quantile regression loss too.
        """
        with params.to_module(cost_qnet):
            quantiles = cost_qnet(obs, action)
            if target_quantiles is not None:
                loss_val, _ = compute_quantile_regression_loss(quantiles, target_quantiles)
                return quantiles, loss_val
            return quantiles, quantiles.new_zeros(())

    def compute_quantile_regression_loss(current_quantiles, target_quantiles):
        """
        Compute quantile regression loss (|.| variant of the asymmetric loss).

        Args:
            current_quantiles: (num_slices, n_quantiles) - current Q-value quantiles
            target_quantiles: (num_slices, n_quantiles) - target Q-value quantiles

        Returns:
            loss: scalar loss value
            loss_per_sample: (num_slices,) per-sample losses
        """
        # diff[i, j, k] = target[i, k] - current[i, j]
        # Shape: (num_slices, n_quantiles, n_quantiles)
        diff = target_quantiles.unsqueeze(-2) - current_quantiles.unsqueeze(-1)

        # ρ_τ(u) = |τ - 𝟙_{u<0}| * |u|
        loss_per_sample = (diff.abs() * (tau_hat_reshaped - (diff.detach() < 0).float()).abs()).mean(2).mean(1)

        loss = loss_per_sample.mean()
        return loss, loss_per_sample
    
    def update_main(data):
        """
        Update reward and cost critics using distributional quantile regression.
        
        NOTE: N-step returns are computed by MultiStepTransform attached to the replay buffer.
        MultiStepTransform stores data with nested structure:
        - data["observations"] = s_t (current state)
        - data["actions"] = a_t (current action)
        - data["next"]["observations"] = s_{t+n} (n steps ahead)
        - data["next"]["rewards"] = sum_{k=0}^{n-1} gamma^k * r_{t+k} (n-step accumulated reward)
        - data["next"]["costs"] = sum_{k=0}^{n-1} gamma^k * c_{t+k} (n-step accumulated cost)
        - data["next"]["done"] = done flag
        - data["gamma"] = gamma^n (discount factor to the power of n_step)
        - data["steps_to_next_obs"] = actual steps to next obs (may be < n if episode ended)
        
        For each quantile τ_i, the critic loss is:

        Reward Critic (Distributional Bellman with n-step returns):
        $$L_Q = \mathbb{E}_{(s,a,r^{(n)},s_{t+n}) \sim \mathcal{D}} \left[ \sum_{i=1}^N \rho_{\tau_i}(\delta_i) \right]$$

        where:
        $$\delta_i = r^{(n)} + \gamma^n (1 - d) \cdot \left(Z_{\theta'}(s_{t+n}, \tilde{a}') - Z_\theta(s, a)_{\tau_i}\right)$$

        $$\rho_{\tau}(u) = |\tau - \mathbb{1}_{u < 0}| \cdot |u|$$

        Cost Critics (one distributional Bellman backup per cost dimension j, with n-step):

        $$L_C = \sum_{j=1}^{C} \mathbb{E}_{(s,a,c_j^{(n)},s_{t+n}) \sim \mathcal{D}} \left[ \sum_{i=1}^N \rho_{\tau_i}(\delta^{c,j}_i) \right]$$

        where:
        $$\delta^{c,j}_i = c_j^{(n)} + \gamma^n (1 - d) \cdot Z^{c,j}_{\theta'}(s_{t+n}, \tilde{a}')_{\tau_i} - Z^{c,j}_\theta(s, a)_{\tau_i}$$

        Notation:
        - $r^{(n)}$ = n-step return (pre-computed by MultiStepTransform)
        - $Z_\theta(s,a)$ = quantile function (outputs N quantiles)
        - $\tau_i$ = quantile level (i-th quantile)
        - $\tilde{a}'$ = deterministic action from target policy at next state
        - $N$ = number of quantiles (n_quantiles)
        - $n$ = n-step lookahead
        """
        q_optimizer.zero_grad()
        cost_optimizer.zero_grad()

        alpha = log_alpha.exp().detach()

        # Use gamma from MultiStepTransform (accounts for early episode termination)
        # When episode ends after k < n steps, data["gamma"] = gamma^k (not gamma^n)
        gamma_n = data["gamma"]

        with torch.no_grad():
            next_obs = data["next"]["observations"]
            next_rewards = data["next"]["rewards"]
            next_costs = data["next"]["costs"]
            next_done = data["next"]["done"]

            batch_size = next_obs.shape[0]

            # Deterministic target action (temperature=0 => use mean of policy).
            next_state_actions, next_state_log_pi, _, _, _, _ = target_actor.get_action(
                next_obs, temperature=0.0
            )
            # Get quantiles from both critics and select the one with lower expected value (SAC-style)
            next_quantiles_both, _ = torch.vmap(batched_qf, (0, None, None))(
                qnet_target, next_obs, next_state_actions
            )
            expected_values = next_quantiles_both.mean(dim=-1)
            min_critic_idx = expected_values.argmin(dim=0)
            next_quantiles = next_quantiles_both[min_critic_idx, torch.arange(batch_size), :]

            next_quantiles = next_quantiles - alpha * next_state_log_pi

            # (num_costs, batch_size, n_quantiles)
            cost_next_quantiles, _ = torch.vmap(batched_cost_qf, (0, None, None))(
                cost_qnet_target, next_obs, next_state_actions
            )

            # Compute target quantiles: r^{(n)} (from MultiStepTransform) + gamma^n * next_quantiles
            # next_rewards already contains the n-step accumulated rewards from MultiStepTransform
            # gamma_n has shape (batch_size,), need to unsqueeze for broadcasting with quantiles
            target_quantiles = next_rewards.unsqueeze(-1) + (~next_done.unsqueeze(-1)).float() * gamma_n.unsqueeze(-1) * next_quantiles

            # Cost critic targets (also distributional with n-step), one per cost dimension.
            # next_costs is (batch_size, num_costs); transpose so the cost dimension leads.
            target_cost_quantiles = next_costs.t().unsqueeze(-1) + (~next_done.unsqueeze(-1)).float() * gamma_n.unsqueeze(-1) * cost_next_quantiles

        # Current quantiles from both critics and compute loss for each
        # Use vmap to compute loss for both qf1 and qf2
        current_quantiles, qf_losses = torch.vmap(batched_qf, (0, None, None, None))(
            qnet_params, data["observations"], data["actions"], target_quantiles
        )
        qf_loss = qf_losses.sum()  # Sum losses from both critics

        qf_loss.backward()
        q_grad_norm = torch.nn.utils.clip_grad_norm_(qnet.parameters(), args.qf_max_grad_norm) if args.qf_max_grad_norm is not None else torch.tensor(0.0)
        q_optimizer.step()

        # Cost critic loss (quantile regression), computed inside vmap for every cost dimension
        current_cost_quantiles, cost_losses = torch.vmap(batched_cost_qf, (0, None, None, 0))(
            cost_qnet_params, data["observations"], data["actions"], target_cost_quantiles
        )
        cost_loss = cost_losses.sum()  # Sum losses over cost dimensions
        cost_loss.backward()
        cost_q_grad_norm = torch.nn.utils.clip_grad_norm_(cost_qnet.parameters(), args.cost_qf_max_grad_norm) if args.cost_qf_max_grad_norm is not None else torch.tensor(0.0)
        cost_optimizer.step()
        
        # Compute expected Q-values (mean of quantiles)
        with torch.no_grad():
            expected_q = current_quantiles.mean(dim=1).mean()
            # (num_costs,) so every constraint's critic can be tracked separately
            expected_cost_q_per_dim = current_cost_quantiles.mean(dim=(1, 2))
            expected_cost_q = expected_cost_q_per_dim.mean()

        return TensorDict(
            qf_loss=qf_loss.detach(),
            cost_loss=cost_loss.detach(),
            q_grad_norm=q_grad_norm.detach(),
            cost_q_grad_norm=cost_q_grad_norm.detach(),
            expected_q=expected_q.detach(),
            expected_cost_q=expected_cost_q.detach(),
            expected_cost_q_per_dim=expected_cost_q_per_dim.detach(),
        )

    # Capture stock configuration in closure to avoid Python conditionals during CUDA graph capture.
    use_cost_stock = args.add_cost_stock and cost_stock_index is not None
    cost_stock_start = cost_stock_index if use_cost_stock else 0  # first of the num_costs stock entries
    cost_stock_end = cost_stock_start + num_costs

    def update_pol(data, constants):
        """
        Update the actor with a SAC-style entropy-regularized objective and safety costs.

        Reward Q-value:
        $$U(s, a) = \mathbb{E}_{\tau}[Z_\theta(s, a)_\tau]$$

        Cost Q-value per cost dimension j (stock-aware if enabled):
        - With cost stock:
          $$U^c_j(s, a) = \mathbb{E}_{\tau}[\max(0, Z^{c,j}_\theta(s, a)_\tau + \text{stock}_{c,j} \cdot \text{normalizer}_{c,j})]$$
        - Without cost stock:
          $$U^c_j(s, a) = \mathbb{E}_{\tau}[Z^{c,j}_\theta(s, a)_\tau]$$

        SAC-style actor loss with C constraints:
        $$L_\pi = \mathbb{E}_{s \sim \mathcal{D}} \left[
            \frac{\alpha \log \pi(\tilde{a}|s) - U(s,\tilde{a}) + \sum_{j=1}^{C} \lambda_j U^c_j(s,\tilde{a}) + L_{\text{reg}}(s)}
                 {1 + \sum_{j=1}^{C} \lambda_j}
        \right]$$

        Policy regularization:
        $$L_{\text{reg}}(s) = \beta (\|\mu(s)\|^2 + \|\log \sigma(s)\|^2)$$

        Notation:
        - $\tilde{a}$ = action sampled via reparameterization: $\tilde{a} = \tanh(\mu(s) + \sigma(s) \odot \epsilon)$
        - $\alpha$ = entropy coefficient
        - $\beta$ = regularization weight (policy_reg_weight)
        - $\lambda_j$ = Lagrange multiplier of constraint j (learned to enforce $J^c_j \leq d_j$).
          With a contextual multiplier it is $\lambda_j(c_{0,j})$, i.e. each constraint is
          priced according to its own initial budget.
        """
        actor_optimizer.zero_grad()
        pi, log_pi, _, _, mean_raw, log_std = actor.get_action(data["observations"])
        
        # Entropy tuning
        alpha_loss = torch.tensor(0.0, device=device)
        alpha = log_alpha.exp()
        
        if args.autotune:
             alpha_optimizer.zero_grad()
             alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy).detach()).mean()
             alpha_loss.backward()
             alpha_optimizer.step()
             alpha = log_alpha.exp()

        # Get quantiles from both critics and select the one with lower expected value (SAC-style)
        qf_pi_quantiles_both, _ = torch.vmap(batched_qf, (0, None, None))(
            qnet_params.data, data["observations"], pi
        )
        # Compute expected values: (2, batch_size)
        expected_values = qf_pi_quantiles_both.mean(dim=-1)
        # Select the critic index with minimum expected value: (batch_size,)
        min_critic_idx = expected_values.argmin(dim=0)
        # Gather quantiles from the selected critic: (batch_size, n_quantiles)
        batch_size = pi.shape[0]
        qf_pi_quantiles = qf_pi_quantiles_both[min_critic_idx, torch.arange(batch_size), :]
        
        qf_pi = qf_pi_quantiles.mean(dim=-1, keepdim=True)

        # (num_costs, batch_size, n_quantiles)
        cost_q_pi_quantiles, _ = torch.vmap(batched_cost_qf, (0, None, None))(
            cost_qnet_params, data["observations"], pi
        )

        if use_cost_stock:
            # Extract the per-dimension cost stocks from observations
            cost_stock = data["observations"][:, cost_stock_start:cost_stock_end]  # (batch_size, num_costs)
            # Transpose to (num_costs, batch_size, 1) to broadcast over quantiles
            cost_stock_value = (constants["cost_normalizer"] * cost_stock).t().unsqueeze(-1)

            # Add cost stock to quantiles and apply utility function
            if args.cost_utility_type == "abs":
                cost_utility_quantiles = cost_q_pi_quantiles + cost_stock_value  # (num_costs, batch_size, n_quantiles)
                cost_utility_quantiles = torch.clamp(cost_utility_quantiles, min=0.0)  # Utility function
            elif args.cost_utility_type == "mean":
                cost_utility_quantiles = cost_q_pi_quantiles + cost_stock_value  # (num_costs, batch_size, n_quantiles)
            else:
                raise ValueError(f"Invalid cost_utility_type: {args.cost_utility_type}")
            cost_q_pi = cost_utility_quantiles.mean(dim=-1).t()  # (batch_size, num_costs)
        else:
            # Standard cost Q-values without stock
            cost_q_pi = cost_q_pi_quantiles.mean(dim=-1).t()  # (batch_size, num_costs)

        # Get Lagrangian multipliers (one per cost dimension, optionally contextual)
        if "contextual_lagrangian" in constants.keys():
            # Lookup λ(c₀) for each sample and detach it from the actor update.
            # Lambda parameters are updated separately by lambda_optimizer.
            c0 = data["initial_cost_stock"]  # (batch_size, num_costs)
            lagrangian_multiplier_per_sample = constants["contextual_lagrangian"](c0).detach()  # (batch_size, num_costs)
            lagrangian_multiplier_mean = lagrangian_multiplier_per_sample.mean()
        else:
            # Fixed λ vector: same for all samples
            lagrangian_multiplier_per_sample = constants["lagrangian_multiplier"]  # (num_costs,)
            lagrangian_multiplier_mean = lagrangian_multiplier_per_sample.mean()

        # Keep the mean/std regularizer separate from the SAC objective for logging.
        mean_reg_loss = args.policy_reg_weight * torch.mean(mean_raw ** 2)
        std_reg_loss = args.policy_reg_weight * torch.mean(log_std ** 2)
        policy_reg_loss = mean_reg_loss + std_reg_loss
        
        # Combine policy, entropy, and safety losses. Contextual mode uses each
        # sample's own λ(c₀), while scalar mode uses one shared multiplier.
        if "contextual_lagrangian" in constants.keys():
            per_sample_policy_loss = alpha.detach() * log_pi - qf_pi  # (batch_size, 1)
            per_sample_reg = args.policy_reg_weight * (mean_raw ** 2 + log_std ** 2).sum(dim=-1, keepdim=True)  # (batch_size, 1)
            # Sum the priced constraints: Σ_j λ_j(c₀_j) · U^c_j
            per_sample_cost = (lagrangian_multiplier_per_sample * cost_q_pi).sum(dim=-1, keepdim=True)  # (batch_size, 1)

            numerator = per_sample_policy_loss + per_sample_reg + per_sample_cost
            denominator = 1 + lagrangian_multiplier_per_sample.sum(dim=-1, keepdim=True)
            actor_loss = (numerator / denominator).mean()
        else:
            policy_loss = (alpha.detach() * log_pi - qf_pi).mean()
            cost_term = (lagrangian_multiplier_per_sample * cost_q_pi.mean(dim=0)).sum()
            actor_loss = (policy_loss + policy_reg_loss + cost_term) / (1 + lagrangian_multiplier_per_sample.sum())

        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), args.actor_max_grad_norm) if args.actor_max_grad_norm is not None else torch.tensor(0.0)
        actor_optimizer.step()
        # Return losses
        entropy_const = 0.5 * (np.log(2 * np.pi) + 1.0)
        policy_entropy = (log_std + entropy_const).sum(dim=1, keepdim=True).mean().detach()
        policy_avg_std = log_std.exp().mean().detach()
        policy_avg_mean = mean_raw.mean().detach()
        return TensorDict(
            actor_loss=actor_loss.detach(),
            policy_entropy=policy_entropy,
            policy_avg_std=policy_avg_std,
            policy_avg_mean=policy_avg_mean,
            actor_grad_norm=actor_grad_norm.detach(),
            lagrangian_multiplier_mean=lagrangian_multiplier_mean,
            # Loss components for analysis
            qf_pi_mean=qf_pi.mean().detach(),
            cost_q_pi_mean=cost_q_pi.mean().detach(),
            cost_q_pi_per_dim=cost_q_pi.mean(dim=0).detach(),  # (num_costs,)
            policy_reg_loss=policy_reg_loss.detach(),
            alpha=alpha.detach(),
            alpha_loss=alpha_loss.detach(),
        )

    if args.compile:
        mode = None  # "reduce-overhead" if not args.cudagraphs else None
        update_main = torch.compile(update_main, mode=mode)
        update_pol = torch.compile(update_pol, mode=mode)
        policy = torch.compile(policy, mode=mode)

    if args.cudagraphs:
        update_main = CudaGraphModule(update_main, in_keys=[], out_keys=[])
        update_pol = CudaGraphModule(update_pol, in_keys=[], out_keys=[])
        policy = CudaGraphModule(policy)

    obs, _ = envs.reset(seed=args.seed)
    obs = torch.as_tensor(obs, device=device, dtype=torch.float)
    pbar = tqdm.tqdm(range(args.total_timesteps))
    start_time = None
    measure_burnin = 0
    max_ep_ret = -float("inf")
    avg_returns = deque(maxlen=args.lagrange_buffer_size)
    avg_costs = deque(maxlen=args.lagrange_buffer_size)
    avg_lengths = deque(maxlen=args.lagrange_buffer_size)
    avg_objective = deque(maxlen=args.lagrange_buffer_size)
    max_ep_cost = np.zeros(num_costs, dtype=np.float32)
    # The env's own scalar cost, tracked alongside the per-dimension vector so this
    # run can be compared directly against a single-cost (ucp.py) baseline.
    avg_native_costs = deque(maxlen=args.lagrange_buffer_size)
    ep_native_costs = np.zeros(envs.num_envs, dtype=np.float32)
    ep_returns = np.zeros(envs.num_envs, dtype=np.float32)
    ep_costs = np.zeros((envs.num_envs, num_costs), dtype=np.float32)
    ep_lengths = np.zeros(envs.num_envs, dtype=np.int32)
    # Discounted episode returns and costs
    discounted_ep_returns = np.zeros(envs.num_envs, dtype=np.float32)
    discounted_ep_costs = np.zeros((envs.num_envs, num_costs), dtype=np.float32)
    discount_powers = np.ones(envs.num_envs, dtype=np.float32)  # Track gamma^t for each env (start at 1.0)
    avg_discounted_returns = deque(maxlen=args.lagrange_buffer_size)
    avg_discounted_costs = deque(maxlen=args.lagrange_buffer_size)
    
    # Track the initial cost stock c₀ for each environment. Contextual λ updates
    # use c₀ with the completed episode cost to measure budget violation.
    if args.sample_initial_cost_stock and cost_stock_index is not None:
        # Initialize with the initial observation's cost stocks (already normalized)
        initial_cost_stocks = obs[:, cost_stock_index:cost_stock_index + num_costs].cpu().numpy() * cost_normalizers_np
    else:
        initial_cost_stocks = np.tile(initial_cost_stocks_np, (envs.num_envs, 1))

    # Sliding window of completed trajectory (c₀, episodic_cost) pairs for contextual λ updates.
    # Bounded deque: each update uses the most recent maxlen trajectories, preventing stale signal
    # from piling up indefinitely (e.g. during lagrange_warmup_steps).
    trajectory_cost_buffer = deque(maxlen=args.lagrange_buffer_size)
    
    desc = ""

    constants = TensorDict({
        "cost_normalizer": cost_normalizers_t,  # (num_costs,)
    }, batch_size=[], device=device)
    gradient_step_counter = 0  # Track actor gradient steps.

    for global_step in pbar:
        if global_step == args.measure_burnin + args.learning_starts:
            start_time = time.time()
            measure_burnin = global_step

        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            obs_dict = TensorDict({"observation": obs}, batch_size=obs.shape[0])
            actions = policy(obs_dict)["action"]
            actions = actions.cpu().numpy()

        next_obs, rewards, costs, terminations, truncations, infos = envs.step(actions)

        rewards = np.asarray(rewards, dtype=np.float32)
        costs = np.asarray(costs, dtype=np.float32)
        terminations = np.asarray(terminations, dtype=bool)
        truncations = np.asarray(truncations, dtype=bool)

        dones_np = np.logical_or(terminations, truncations)

        ep_returns += rewards
        ep_costs += costs
        # Falls back to the vector sum if the wrapper stack did not supply it.
        ep_native_costs += np.asarray(infos.get("native_cost", costs.sum(axis=1)), dtype=np.float32)
        ep_lengths += 1

        # Track discounted returns and costs
        discounted_ep_returns += discount_powers * rewards
        discounted_ep_costs += discount_powers[:, None] * costs

        done_indices = np.where(dones_np)[0]
        for idx in done_indices:
            ep_return = float(ep_returns[idx])
            ep_cost = ep_costs[idx].copy()          # (num_costs,)
            ep_length = float(ep_lengths[idx])
            discounted_return = float(discounted_ep_returns[idx])
            discounted_cost = discounted_ep_costs[idx].copy()

            avg_returns.append(ep_return)
            avg_costs.append(ep_cost)
            avg_native_costs.append(float(ep_native_costs[idx]))
            avg_lengths.append(ep_length)
            avg_discounted_returns.append(discounted_return)
            avg_discounted_costs.append(discounted_cost)
            # Compute objective using appropriate lambda vector
            if args.sample_initial_cost_stock and contextual_lagrangian is not None:
                c0_tensor = torch.as_tensor(initial_cost_stocks[idx], device=device).unsqueeze(0)
                lambda_val = contextual_lagrangian(c0_tensor)[0].detach().cpu().numpy()
            else:
                lambda_val = lagrangian_multiplier.detach().cpu().numpy()
            avg_objective.append(ep_return - float(np.dot(lambda_val, ep_cost)))
            max_ep_ret = max(max_ep_ret, ep_return)
            max_ep_cost = np.maximum(max_ep_cost, ep_cost)

            # Store the completed trajectory for the next contextual λ update.
            if args.sample_initial_cost_stock and contextual_lagrangian is not None:
                trajectory_cost_buffer.append((initial_cost_stocks[idx].copy(), ep_cost))

            ep_returns[idx] = 0.0
            ep_costs[idx] = 0.0
            ep_native_costs[idx] = 0.0
            ep_lengths[idx] = 0
            discounted_ep_returns[idx] = 0.0
            discounted_ep_costs[idx] = 0.0
            discount_powers[idx] = 1.0  # Reset for next episode
        
        # Update discount powers for next step (gamma^(t+1) = gamma^t * gamma)
        discount_powers = np.where(dones_np, 1.0, discount_powers * args.gamma)

        if avg_returns:
            mean_ret = float(np.mean(avg_returns))
            mean_cost_vec = np.mean(np.stack(avg_costs), axis=0) if avg_costs else np.zeros(num_costs)
            mean_len = float(np.mean(avg_lengths)) if avg_lengths else 0.0
            mean_objective = float(np.mean(avg_objective)) if avg_objective else 0.0
            cost_str = "/".join(f"{c:.2f}" for c in mean_cost_vec)
            desc = (
                f"global_step={global_step}, return={mean_ret: 4.2f} (max={max_ep_ret: 4.2f}), "
                f"cost=[{cost_str}], "
                f"len={mean_len: 4.1f}, objective={mean_objective: 4.2f}"
            )

        next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float)
        real_next_obs = next_obs.clone()
        if "final_observation" in infos:
            final_obs = infos["final_observation"]
            for idx, trunc in enumerate(truncations):
                if trunc and final_obs[idx] is not None:
                    final_obs_idx = np.asarray(final_obs[idx], dtype=np.float32)
                    real_next_obs[idx] = torch.as_tensor(final_obs_idx, device=device, dtype=torch.float)
        
        rewards_tensor = torch.as_tensor(rewards, device=device, dtype=torch.float)
        costs_tensor = torch.as_tensor(costs, device=device, dtype=torch.float)
        # Use terminations (not dones) for bootstrapping: only true episode ends stop bootstrapping
        terminations_tensor = torch.as_tensor(terminations, device=device, dtype=torch.bool)

        initial_cost_stocks_tensor = torch.as_tensor(initial_cost_stocks, device=device, dtype=torch.float)

        # MultiStepTransform (attached to rb) will automatically compute n-step returns.
        transition = TensorDict(
            {
                "observations": obs,
                "actions": torch.as_tensor(actions, device=device, dtype=torch.float),
                "initial_cost_stock": initial_cost_stocks_tensor,  # c₀ for contextual λ(c₀)
                "next": TensorDict(
                    {
                        "observations": real_next_obs,
                        "rewards": rewards_tensor,
                        "costs": costs_tensor,
                        "done": terminations_tensor,
                    },
                    batch_size=obs.shape[0],
                    device=device,
                ),
            },
            batch_size=obs.shape[0],
            device=device,
        )
        transition = transition.unsqueeze(-1)  # expose time dimension for MultiStepTransform
        rb.extend(transition)
        
        # Update initial_cost_stocks for environments that just reset
        # The next observation for reset environments contains the new initial cost stocks
        if args.sample_initial_cost_stock and cost_stock_index is not None:
            for idx in done_indices:
                # Get the new initial cost stocks from next_obs (which is now the reset observation)
                initial_cost_stocks[idx] = (
                    next_obs[idx, cost_stock_index:cost_stock_index + num_costs].cpu().numpy() * cost_normalizers_np
                )

        obs = next_obs
        
        # Sample from replay buffer (will have n-step returns computed by MultiStepTransform)
        if rb._storage._len >= args.batch_size:
            sample = rb.sample(args.batch_size)
            batch_shape = sample.batch_size
            if len(batch_shape) > 1 and batch_shape[-1] == 1:
                sample = sample.squeeze(-1)
            
            data = sample
        else:
            data = None

        if global_step > args.learning_starts and data is not None:
            out_main = update_main(data)
            
            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(args.policy_frequency):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    # Set Lagrangian multiplier in constants (scalar or contextual)
                    if args.sample_initial_cost_stock and contextual_lagrangian is not None:
                        constants["contextual_lagrangian"] = contextual_lagrangian
                    else:
                        constants["lagrangian_multiplier"] = lagrangian_multiplier.detach()
                    out_main.update(update_pol(data, constants))
                    
                    gradient_step_counter += 1
            # Lagrange Multiplier Update — runs every step (decoupled from policy_frequency).
            # For contextual λ(c₀): uses the sliding window of recently completed trajectories.
            # For scalar λ: uses the rolling mean episodic cost.
            # Updating every step (when buffer is non-empty) maximises update frequency
            # without tying it to the actor update cadence.
            if avg_costs and global_step > max(args.learning_starts, args.lagrange_warmup_steps):
                """
                Lagrange Multiplier Update for Safety Constraints (one per cost dimension):

                For a fixed λ vector:
                $$\lambda_{j,k+1} = \text{proj}_{[0, \infty)} \left[ \lambda_{j,k} + \eta_\lambda \cdot (J^c_j(\pi_k) - d_j) \right]$$

                For contextual λ_j(c₀_j):
                The constraint is: episodic_cost_j + c₀_j ≤ 0 (i.e., cost_stock_final_j ≤ 0)
                The violation is: episodic_cost_j + c₀_j
                We use actual trajectory costs to update λ_j(c₀_j) for each completed trajectory.
                Each dimension is updated from its own violation, so a satisfied constraint
                does not raise the price of a violated one (and vice versa).
                """
                if args.sample_initial_cost_stock and contextual_lagrangian is not None:
                    # Update each active (dimension, bin) from recently completed trajectories.
                    if len(trajectory_cost_buffer) > 0:
                        # Convert buffer to tensors: both (n_trajectories, num_costs)
                        c0_list, ep_cost_list = zip(*trajectory_cost_buffer)
                        c0_batch = torch.tensor(np.stack(c0_list), device=device, dtype=torch.float32)
                        ep_cost_batch = torch.tensor(np.stack(ep_cost_list), device=device, dtype=torch.float32)

                        # Get λ(c₀) for each trajectory's initial cost stocks.
                        lambda_vals = contextual_lagrangian(c0_batch)  # (n_trajectories, num_costs)

                        # Constraint violation: episodic_cost + c₀ (should be ≤ 0)
                        # Since c₀ is negative (e.g., -25), violation > 0 means episodic_cost > |c₀|
                        if args.lagrangian_utility_type == "abs":
                            # For utility-based, we want to penalize based on how much the cost exceeds the threshold, so we can use the raw violation
                            violation = torch.clamp(ep_cost_batch + c0_batch, min=0.0) - args.lagrangian_utility_epsilon  # (n_trajectories, num_costs)
                        elif args.lagrangian_utility_type == "mean":
                            # For mean-based, we can use the raw violation which can be positive or negative, allowing λ to adjust in either direction
                            violation = ep_cost_batch + c0_batch  # (n_trajectories, num_costs)

                        # Minimizing -λ(c₀) * violation performs gradient ascent on
                        # λ for bins whose recent trajectories exceed their budget.
                        # Summing over dimensions keeps each dimension's gradient at the
                        # same scale as the single-cost case.
                        lambda_loss = -(lambda_vals * violation).sum(dim=-1).mean()

                        lambda_optimizer.zero_grad()
                        lambda_loss.backward()
                        lambda_optimizer.step()
                        # Bounds are applied inside ContextualLagrangian.forward().
                else:
                    # Fixed-budget Lagrangian update, per cost dimension
                    recent_costs = np.stack(avg_costs)  # (n_trajectories, num_costs)
                    if args.lagrangian_utility_type == "abs":
                        violation = np.mean(np.maximum(initial_cost_stocks_np + recent_costs, 0.0), axis=0) - args.lagrangian_utility_epsilon
                    elif args.lagrangian_utility_type == "mean":
                        violation = np.mean(initial_cost_stocks_np + recent_costs, axis=0)

                    violation_t = torch.as_tensor(violation, device=device, dtype=torch.float32)
                    lambda_loss = -(lagrangian_multiplier * violation_t).sum()
                    lambda_optimizer.zero_grad()
                    lambda_loss.backward()
                    lambda_optimizer.step()

                    with torch.no_grad():
                        lower = args.lagrangian_lower_bound
                        upper = args.lagrangian_upper_bound if args.lagrangian_upper_bound is not None else float("inf")
                        lagrangian_multiplier.clamp_(lower, upper)

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                # lerp is defined as x' = x + w (y-x), which is equivalent to x' = (1-w) x + w y
                qnet_target.lerp_(qnet_params.data, args.q_tau)
                cost_qnet_target.lerp_(cost_qnet_params.data, args.c_tau)
            if global_step % args.target_policy_update_freq == 0:
                target_actor_params.lerp_(actor_params.data, args.p_tau)

            if global_step % 100 == 0 and start_time is not None:
                speed = (global_step - measure_burnin) / (time.time() - start_time)
                pbar.set_description(f"{speed: 4.4f} sps, " + desc)
                with torch.no_grad():
                    # Get lagrange multiplier for logging (mean over dimensions / bins)
                    if args.sample_initial_cost_stock and contextual_lagrangian is not None:
                        lambda_for_log = out_main["lagrangian_multiplier_mean"].item() if "lagrangian_multiplier_mean" in out_main.keys() else contextual_lagrangian.get_all_lambdas().mean().item()
                    else:
                        lambda_for_log = lagrangian_multiplier.mean().item()

                    # Per-dimension episodic costs; the scalar chart keeps the total across constraints.
                    ep_cost_per_dim = np.mean(np.stack(avg_costs), axis=0) if avg_costs else np.zeros(num_costs)
                    disc_cost_per_dim = np.mean(np.stack(avg_discounted_costs), axis=0) if avg_discounted_costs else np.zeros(num_costs)

                    logs = {
                        "charts/episodic_return": torch.tensor(list(avg_returns)).mean() if avg_returns else torch.tensor(0.0),
                        "charts/episodic_cost": torch.tensor(float(ep_cost_per_dim.sum())),
                        # The env's own scalar cost — the number a single-cost baseline reports
                        "charts/episodic_native_cost": torch.tensor(list(avg_native_costs)).mean() if avg_native_costs else torch.tensor(0.0),
                        "charts/episodic_length": torch.tensor(list(avg_lengths)).mean() if avg_lengths else torch.tensor(0.0),
                        "charts/episodic_objective": torch.tensor(list(avg_objective)).mean() if avg_objective else torch.tensor(0.0),
                        "charts/discounted_return": torch.tensor(list(avg_discounted_returns)).mean() if avg_discounted_returns else torch.tensor(0.0),
                        "charts/discounted_cost": torch.tensor(float(disc_cost_per_dim.sum())),
                        "losses/actor_loss": out_main["actor_loss"].mean() if "actor_loss" in out_main.keys() else torch.tensor(0.0),
                        "losses/qf_loss": out_main["qf_loss"].mean(),
                        "losses/cost_loss": out_main["cost_loss"].mean(),
                        "losses/lagrange_multiplier": lambda_for_log,
                        "losses/alpha": out_main["alpha"] if "alpha" in out_main.keys() else torch.tensor(0.0),
                        "losses/alpha_loss": out_main["alpha_loss"] if "alpha_loss" in out_main.keys() else torch.tensor(0.0),
                        "charts/SPS": speed,
                        "policy/entropy": out_main["policy_entropy"] if "policy_entropy" in out_main.keys() else torch.tensor(0.0),
                        "policy/avg_std": out_main["policy_avg_std"] if "policy_avg_std" in out_main.keys() else torch.tensor(0.0),
                        "policy/avg_mean": out_main["policy_avg_mean"] if "policy_avg_mean" in out_main.keys() else torch.tensor(0.0),
                        "policy/gradient_steps": gradient_step_counter,
                        # Actor loss components
                        "loss_components/qf_pi_mean": out_main["qf_pi_mean"] if "qf_pi_mean" in out_main.keys() else torch.tensor(0.0),
                        "loss_components/cost_q_pi_mean": out_main["cost_q_pi_mean"] if "cost_q_pi_mean" in out_main.keys() else torch.tensor(0.0),
                        "loss_components/policy_reg_loss": out_main["policy_reg_loss"] if "policy_reg_loss" in out_main.keys() else torch.tensor(0.0),
                        # Gradient norms
                        "grad_norms/actor": out_main["actor_grad_norm"] if "actor_grad_norm" in out_main.keys() else torch.tensor(0.0),
                        "grad_norms/q_critic": out_main["q_grad_norm"],
                        "grad_norms/cost_critic": out_main["cost_q_grad_norm"],
                        # Expected Q-values from critics
                        "q_values/expected_reward_q": out_main["expected_q"] if "expected_q" in out_main.keys() else torch.tensor(0.0),
                        "q_values/expected_cost_q": out_main["expected_cost_q"] if "expected_cost_q" in out_main.keys() else torch.tensor(0.0),
                    }

                    # Per-cost-dimension charts
                    for j, name in enumerate(cost_names):
                        logs[f"charts_per_cost/episodic_cost_{name}"] = torch.tensor(float(ep_cost_per_dim[j]))
                        logs[f"charts_per_cost/discounted_cost_{name}"] = torch.tensor(float(disc_cost_per_dim[j]))
                        logs[f"charts_per_cost/max_episodic_cost_{name}"] = torch.tensor(float(max_ep_cost[j]))
                        if "cost_q_pi_per_dim" in out_main.keys():
                            logs[f"loss_components_per_cost/cost_q_pi_{name}"] = out_main["cost_q_pi_per_dim"][j]
                        if "expected_cost_q_per_dim" in out_main.keys():
                            logs[f"q_values_per_cost/expected_cost_q_{name}"] = out_main["expected_cost_q_per_dim"][j]
                        if args.sample_initial_cost_stock and contextual_lagrangian is not None:
                            logs[f"losses_per_cost/lagrange_multiplier_{name}"] = (
                                contextual_lagrangian.get_all_lambdas()[j].mean()
                            )
                        else:
                            logs[f"losses_per_cost/lagrange_multiplier_{name}"] = lagrangian_multiplier[j]

                    # Log to TensorBoard
                    if args.use_tb:
                        for key, value in logs.items():
                            if isinstance(value, torch.Tensor):
                                writer.add_scalar(key, value.item(), global_step)
                            else:
                                writer.add_scalar(key, value, global_step)

                        # Log per-bin lambda values for contextual Lagrangian, per cost dimension
                        if args.sample_initial_cost_stock and contextual_lagrangian is not None:
                            all_lambdas = contextual_lagrangian.get_all_lambdas()   # (num_costs, num_bins)
                            bin_centers = contextual_lagrangian.get_bin_centers()   # (num_costs, num_bins)
                            for j, name in enumerate(cost_names):
                                for c0, lam in zip(bin_centers[j].tolist(), all_lambdas[j].tolist()):
                                    writer.add_scalar(f"contextual_lambda/{name}/c0_{c0:.1f}", lam, global_step)

                if args.track:
                    wandb_logs = {
                        "speed": speed,
                        "episode_return": logs["charts/episodic_return"].item() if isinstance(logs["charts/episodic_return"], torch.Tensor) else logs["charts/episodic_return"],
                        "episode_cost": logs["charts/episodic_cost"].item() if isinstance(logs["charts/episodic_cost"], torch.Tensor) else logs["charts/episodic_cost"],
                        "episode_length": logs["charts/episodic_length"].item() if isinstance(logs["charts/episodic_length"], torch.Tensor) else logs["charts/episodic_length"],
                        "discounted_return": logs["charts/discounted_return"].item() if isinstance(logs["charts/discounted_return"], torch.Tensor) else logs["charts/discounted_return"],
                        "discounted_cost": logs["charts/discounted_cost"].item() if isinstance(logs["charts/discounted_cost"], torch.Tensor) else logs["charts/discounted_cost"],
                        "actor_loss": logs["losses/actor_loss"].item() if isinstance(logs["losses/actor_loss"], torch.Tensor) else logs["losses/actor_loss"],
                        "qf_loss": logs["losses/qf_loss"].item() if isinstance(logs["losses/qf_loss"], torch.Tensor) else logs["losses/qf_loss"],
                        "cost_loss": logs["losses/cost_loss"].item() if isinstance(logs["losses/cost_loss"], torch.Tensor) else logs["losses/cost_loss"],
                        "lagrange_multiplier": logs["losses/lagrange_multiplier"],
                        "policy_entropy": logs["policy/entropy"].item() if isinstance(logs["policy/entropy"], torch.Tensor) else logs["policy/entropy"],
                        "policy_avg_std": logs["policy/avg_std"].item() if isinstance(logs["policy/avg_std"], torch.Tensor) else logs["policy/avg_std"],
                    }
                    wandb.log(wandb_logs, step=global_step)
        
        # Periodic checkpoint saving
        if args.save_model and global_step % args.save_freq == 0 and global_step > 0:
            checkpoint_path = os.path.join(f"{args.dir}/{run_name}", f"checkpoint_step_{global_step}.pt")
            #rb_state = rb.state_dict()
            
            checkpoint_dict = {
                'actor_state_dict': actor.state_dict(),
                'qnet_state_dict': qnet.state_dict(),
                'cost_qnet_state_dict': cost_qnet.state_dict(),
                'qnet_target_state_dict': qnet_target.state_dict(),
                'cost_qnet_target_state_dict': cost_qnet_target.state_dict(),
                'actor_optimizer_state_dict': actor_optimizer.state_dict(),
                'q_optimizer_state_dict': q_optimizer.state_dict(),
                'cost_optimizer_state_dict': cost_optimizer.state_dict(),
                'lagrangian_multiplier': (lagrangian_multiplier.detach().cpu().tolist() if lagrangian_multiplier is not None
                                      else contextual_lagrangian.get_all_lambdas().mean(dim=-1).detach().cpu().tolist()),
                'lambda_optimizer_state_dict': lambda_optimizer.state_dict(),
                'global_step': global_step,
                'args': vars(args),
                #'replay_buffer': rb_state,
                'gradient_step_counter': gradient_step_counter,
                'avg_returns': list(avg_returns),
                'avg_costs': [c.tolist() for c in avg_costs],
                'avg_lengths': list(avg_lengths),
                'avg_discounted_returns': list(avg_discounted_returns),
                'avg_discounted_costs': [c.tolist() for c in avg_discounted_costs],
                'avg_objective': list(avg_objective),
                'max_ep_ret': max_ep_ret,
                'max_ep_cost': max_ep_cost.tolist(),
                'num_costs': num_costs,
                'cost_names': cost_names,
            }
            
            # Add contextual Lagrangian state if using it
            if args.sample_initial_cost_stock and contextual_lagrangian is not None:
                checkpoint_dict['contextual_lagrangian_state_dict'] = contextual_lagrangian.state_dict()
                checkpoint_dict['contextual_lambda_optimizer_state_dict'] = lambda_optimizer.state_dict()
            
            torch.save(checkpoint_dict, checkpoint_path)
            
            print(f"\n✓ Checkpoint saved: {checkpoint_path} (step {global_step})")
            
    envs.close()
    # Model saving
    if args.save_model:
        model_save_path = os.path.join(f"{args.dir}/{run_name}", f"model_step_{global_step}.pt")
        
        # Save replay buffer state
        #rb_state = rb.state_dict()
        
        model_dict = {
            'actor_state_dict': actor.state_dict(),
            'qnet_state_dict': qnet.state_dict(),
            'cost_qnet_state_dict': cost_qnet.state_dict(),
            'qnet_target_state_dict': qnet_target.state_dict(),
            'cost_qnet_target_state_dict': cost_qnet_target.state_dict(),
            'actor_optimizer_state_dict': actor_optimizer.state_dict(),
            'q_optimizer_state_dict': q_optimizer.state_dict(),
            'cost_optimizer_state_dict': cost_optimizer.state_dict(),
            'lagrangian_multiplier': (lagrangian_multiplier.detach().cpu().tolist() if lagrangian_multiplier is not None
                                      else contextual_lagrangian.get_all_lambdas().mean(dim=-1).detach().cpu().tolist()),
            'lambda_optimizer_state_dict': lambda_optimizer.state_dict(),
            'global_step': global_step,
            'args': vars(args),
            #'replay_buffer': rb_state,
            'gradient_step_counter': gradient_step_counter,
            'avg_returns': list(avg_returns),
            'avg_costs': [c.tolist() for c in avg_costs],
            'avg_lengths': list(avg_lengths),
            'avg_discounted_returns': list(avg_discounted_returns),
            'avg_discounted_costs': [c.tolist() for c in avg_discounted_costs],
            'avg_objective': list(avg_objective),
            'max_ep_ret': max_ep_ret,
            'max_ep_cost': max_ep_cost.tolist(),
            'num_costs': num_costs,
            'cost_names': cost_names,
        }
        
        # Add contextual Lagrangian state if using it
        if args.sample_initial_cost_stock and contextual_lagrangian is not None:
            model_dict['contextual_lagrangian_state_dict'] = contextual_lagrangian.state_dict()
            model_dict['contextual_lambda_optimizer_state_dict'] = lambda_optimizer.state_dict()
        
        torch.save(model_dict, model_save_path)
        
        print(f"\nModel saved to: {model_save_path}\n")
        print(f"  Replay buffer size: {rb._storage._len}")
        print(f"  Gradient steps: {gradient_step_counter}")
    
    # Evaluation
    if args.evaluation_episodes > 0 and args.save_model:
        # Check if multi-stock evaluation is requested
        if args.eval_initial_cost_stocks is not None and len(args.eval_initial_cost_stocks) > 0:
            # Multi-stock evaluation: evaluate with each initial cost stock value
            eval_results = evaluate_ucp_multicost_multi_stock(
                model_save_path,
                eval_episodes_per_stock=args.evaluation_episodes,
                initial_cost_stocks=args.eval_initial_cost_stocks,
                evaluation_temperature=args.evaluation_temperature,
            )

            if writer is not None:
                # Aggregate results. mean_cost sums the cost vector; mean_native_cost is
                # the env's own scalar cost, i.e. what a single-cost baseline reports.
                writer.add_scalar("eval/mean_return", eval_results['aggregate']['mean_return'], global_step)
                writer.add_scalar("eval/mean_cost", eval_results['aggregate']['mean_cost'], global_step)
                writer.add_scalar("eval/std_return", eval_results['aggregate']['std_return'], global_step)
                writer.add_scalar("eval/std_cost", eval_results['aggregate']['std_cost'], global_step)
                writer.add_scalar("eval/mean_native_cost", eval_results['aggregate']['mean_native_cost'], global_step)
                writer.add_scalar("eval/std_native_cost", eval_results['aggregate']['std_native_cost'], global_step)
                for name, value in zip(eval_results['cost_names'], eval_results['aggregate']['mean_cost_per_dim']):
                    writer.add_scalar(f"eval_per_cost/mean_cost_{name}", value, global_step)

                # Log per-stock results
                for stock, res in eval_results['per_stock_results'].items():
                    stock_tag = f"stock_{stock:.1f}".replace('.', '_').replace('-', 'n')
                    writer.add_scalar(f"eval_per_stock/{stock_tag}/mean_return", res['mean_return'], global_step)
                    writer.add_scalar(f"eval_per_stock/{stock_tag}/mean_cost", res['mean_cost'], global_step)
                    writer.add_scalar(f"eval_per_stock/{stock_tag}/mean_native_cost", res['mean_native_cost'], global_step)
                    for name, value in zip(eval_results['cost_names'], res['mean_cost_per_dim']):
                        writer.add_scalar(f"eval_per_stock/{stock_tag}/mean_cost_{name}", value, global_step)
        else:
            # Standard single-stock evaluation
            eval_results = evaluate_ucp_multicost(model_save_path, args.evaluation_episodes, args.evaluation_temperature)

            if writer is not None:
                writer.add_scalar("eval/mean_return", eval_results['mean_return'], global_step)
                writer.add_scalar("eval/mean_cost", eval_results['mean_cost'], global_step)
                writer.add_scalar("eval/std_return", eval_results['std_return'], global_step)
                writer.add_scalar("eval/std_cost", eval_results['std_cost'], global_step)
                writer.add_scalar("eval/mean_native_cost", eval_results['mean_native_cost'], global_step)
                writer.add_scalar("eval/std_native_cost", eval_results['std_native_cost'], global_step)
                for name, value in zip(eval_results['cost_names'], eval_results['mean_cost_per_dim']):
                    writer.add_scalar(f"eval_per_cost/mean_cost_{name}", value, global_step)
    if writer is not None:
        writer.close()