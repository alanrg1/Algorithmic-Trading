import torch
import torch.nn as nn
from torch.distributions import Normal

EPS = 1e-6


class ActorCritic(nn.Module):
    """
    Actor-Critic Network 
    1. Actor head: outputs mean, with learnable log_std parameter
    2. Critic head: outputs single state-value estimate
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        # TODO: Build the critic network
        # Hint: 2-layer MLP that maps state_dim -> hidden_dim (with ReLU) -> 1 scalar value
        # Use nn.Sequential with nn.Linear and nn.ReLU layers
        self.critic = None  # TODO

        # TODO: Build the actor network
        # Hint: 2-layer MLP that maps state_dim -> hidden_dim (with ReLU) -> action_dim (mean of policy)
        # Use nn.Sequential with nn.Linear and nn.ReLU layers
        self.actor = None  # TODO

        # TODO: Create a learnable log standard deviation parameter
        # Hint: nn.Parameter of shape (1, action_dim), initialized to zeros
        # This represents log(σ) for the Gaussian policy — shared across all states
        self.log_std = None  # TODO

        # Apply weight initialization to all layers
        self.apply(self._init_weights)
    
    def forward(self, state):
        """
        Returns:
            dist: Normal distribution with mean from actor and std from log_std
            value: Tensor of shape (batch_size, 1) — state value estimate
        """
        value = self.critic(state)
        mu = self.actor(state) 

        # Clamp log_std to prevent extreme values 
        log_std = torch.clamp(self.log_std, -5, 2)
        dist = Normal(mu, log_std.exp())
        return dist, value
    
    def get_action_and_log_prob_and_value(self, state):
        """
        Sample an action and compute its log probability and state value.
        
        This uses the "tanh squashing" trick:
        1. Sample from the Gaussian: pre_tanh ~ N(μ, σ)
        2. Squash to bounded range: action = tanh(pre_tanh) ∈ (-1, 1)
        3. Correct the log-probability using the change-of-variables formula (Jacobian)
        
        Args:
            state: Tensor of shape (batch_size, state_dim)
        
        Returns:
            action: Tensor of shape (batch_size, action_dim) — squashed to (-1, 1)
            log_prob: Tensor of shape (batch_size,) — corrected log probability
            value: Tensor of shape (batch_size, 1) — state value estimate
        """
        dist, value = self.forward(state) 

        # Tanh squashing + Jacobian correction 
        # 1. Sample unbounded action from Gaussian
        pre_tanh = dist.sample()              # in range (-inf, inf)
        # 2. Squash to (-1, 1) via tanh
        action = torch.tanh(pre_tanh)         # in range (-1, 1)
        # 3. Correct log-prob for the change of variables (Jacobian determinant)
        #    log π(a) = log N(pre_tanh; μ, σ) - Σ log(1 - tanh²(pre_tanh) + ε)
        log_prob = dist.log_prob(pre_tanh) - torch.log(1 - action.pow(2) + EPS)
        log_prob = log_prob.sum(dim=-1)

        return action, log_prob, value  
    
    def _init_weights(self, m):
        """
        Initialize network weights for stable training.
        
        Args:
            m: A neural network module (called via self.apply())
        """
        # TODO: If m is an nn.Linear layer:
        #   - Initialize weights with orthogonal initialization (gain=1.0)
        #   - Initialize biases to constant 0.0
        # Hint: nn.init.orthogonal_(m.weight, gain=1.0)
        #        nn.init.constant_(m.bias, 0.0)
        pass  # TODO
