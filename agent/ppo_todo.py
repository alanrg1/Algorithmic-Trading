import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from .actor_critic_todo import ActorCritic

EPS = 1e-6

class PPO:
    """
    Proximal Policy Optimization (PPO) with clipped surrogate objective.
    
    Training loop:
    1. Collect trajectories using current policy
    2. Compute advantages using Generalized Advantage Estimation (GAE)
    3. Update policy using clipped surrogate objective
    4. Update value function with MSE loss
    5. Repeat
    """
    
    def __init__(self, env, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                 n_epochs=10, batch_size=64, ent_coef=0.01, vf_coef=0.5):
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef

        # TODO: Create the ActorCritic policy network
        # Hint: ActorCritic takes (state_dim, action_dim) from env.observation_space and env.action_space
        #       state_dim = env.observation_space.shape[0]
        #       action_dim = env.action_space.shape[0]
        self.policy = None  # TODO

        # TODO: Create the Adam optimizer for the policy parameters
        # Hint: optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        #       eps=1e-5 improves numerical stability for RL
        self.optimizer = None  # TODO
    
    def compute_gae(self, rewards, values, dones):
        """
        Compute Generalized Advantage Estimation (GAE).
        
        GAE Formula: A_t = δ_t + (γλ)δ_{t+1} + (γλ)^2 δ_{t+2} + ...
        where δ_t = r_t + γV(s_{t+1}) - V(s_t) is the TD error
        
        Args:
            rewards: List of rewards [r_0, r_1, ..., r_T]
            values: List of value estimates [V(s_0), V(s_1), ..., V(s_T), V(s_{T+1})]
                    Note: len(values) = len(rewards) + 1 (includes bootstrap value)
            dones: List of done flags [done_0, done_1, ..., done_T]
        
        Returns:
            advantages: Normalized advantage estimates (np.array)
            returns: TD(λ) returns for value function training (np.array)
        """

        advantages = []
        gae = 0

        # TODO: Loop backwards through timesteps to compute GAE
        #
        # For each timestep t (from T-1 down to 0):
        #   1. If dones[t] is True, next_value = 0 (terminal state has no future value)
        #      Otherwise, next_value = values[t + 1]
        #   2. Compute TD error: δ_t = r_t + γ * next_value - V(s_t)
        #   3. Accumulate GAE: gae = δ_t + γ * λ * (1 - done_t) * gae
        #      The (1 - done_t) term resets GAE at episode boundaries
        #   4. Insert gae at position 0 of advantages list (since we iterate backwards)
        #
        # Hint: Use reversed(range(len(rewards))) to loop backwards
        # Hint: advantages.insert(0, gae) to prepend
        pass  # TODO: implement the loop

        advantages = np.array(advantages, dtype=np.float32)
        values_arr = np.array(values[:-1], dtype=np.float32)

        # TODO: Compute returns from advantages
        # Hint: returns = advantages + values_arr  (R_t = A_t + V(s_t))
        returns = None  # TODO

        # TODO: Normalize advantages for stable training
        # Hint: advantages = (advantages - mean) / (std + 1e-8)
        pass  # TODO

        return advantages, returns 
    
    def collect_trajectories(self, n_steps):
        """
        Collect trajectories by interacting with the environment.
        
        This is the data collection phase where the agent:
        1. Takes actions using current policy
        2. Observes rewards and next states
        3. Stores all information for later training
        
        Args:
            n_steps: Number of environment steps to collect
        
        Returns:
            Dictionary containing:
                - states: np.array of shape (n_steps, obs_dim)
                - actions: np.array of shape (n_steps, action_dim)
                - log_probs: np.array of shape (n_steps,)
                - rewards: list of length n_steps
                - values: list of length n_steps+1 (includes terminal value)
                - dones: list of length n_steps
        """
        # Step 1: Initialize storage lists for each type of data
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []

        # Step 2: Reset environment and get initial observation
        state, _ = self.env.reset() 
        
        # Step 3: Collect n_steps of experience
        for step in range(n_steps): 

            # TODO: Convert state (numpy) to a PyTorch tensor and add batch dimension
            # Hint: torch.FloatTensor(state).unsqueeze(0) gives shape (1, obs_dim)
            state_tensor = None  # TODO

            # TODO: Get action, log_prob, and value from the policy (no gradient needed!)
            # Hint: Wrap in `with torch.no_grad():` since we're just collecting data
            #       action, log_prob, value = self.policy.get_action_and_log_prob_and_value(state_tensor)
            #       Then convert: action -> numpy [0], log_prob -> scalar item(), value -> numpy [0,0]
            action = None    # TODO
            log_prob = None  # TODO
            value = None     # TODO
            
            # TODO: Take the action in the environment
            # Hint: next_state, reward, done, truncated, info = self.env.step(action)
            next_state, reward, done, truncated, _ = None  # TODO

            # TODO: Store this transition in the lists
            # Append: state, action, log_prob, reward, value, (done or truncated)
            pass  # TODO

            # TODO: Update state; if episode ended (done or truncated), reset the environment
            # Hint: state = next_state; if done or truncated: state, _ = self.env.reset()
            pass  # TODO
            
        # TODO: Get final value estimate for the last state (needed for GAE bootstrap)
        # Hint: Same pattern as above — convert state to tensor, get value with no_grad
        #       Append this final value to the values list
        pass  # TODO
            
        return {
            'states': np.array(states, dtype=np.float32),
            'actions': np.array(actions, dtype=np.float32),
            'log_probs': np.array(log_probs, dtype=np.float32),
            'rewards': rewards,
            'values': values,
            'dones': dones
        }


    def update_policy(self, states, actions, old_log_probs, advantages, returns):
        """
        Update policy using PPO's clipped surrogate objective.
        
        PPO Objective: L^{CLIP}(θ) = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]
        where r(θ) = π_θ(a|s) / π_θ_old(a|s) is the probability ratio
        
        This prevents too large policy updates by clipping the ratio.
        
        Args:
            states: np.array of states
            actions: np.array of actions taken (tanh-squashed, in [-1, 1])
            old_log_probs: np.array of log probabilities from collection phase
            advantages: np.array of advantage estimates
            returns: np.array of TD(λ) returns
        
        Returns:
            Dictionary with training statistics (losses, etc.)
        """
        # Convert data to tensors 
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        dataset = TensorDataset(states, actions, old_log_probs, advantages, returns)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        policy_losses, value_losses, entropy_losses = [], [], []
        for epoch in range(self.n_epochs):
            for batch in dataloader:
                batch_states, batch_actions, batch_old_lp, batch_adv, batch_ret = batch

                # Step 1: Get current distribution and value from policy
                dist, batch_values = self.policy.forward(batch_states)
                
                # Inverse-tanh + Jacobian correction 
                # Actions were tanh-squashed during collection, so we invert to recompute log-probs
                a_clamped = torch.clamp(batch_actions, -1 + EPS, 1 - EPS)
                pre_tanh = torch.atanh(a_clamped)
                batch_log_probs = (dist.log_prob(pre_tanh) - torch.log(1 - a_clamped.pow(2) + EPS)).sum(dim=-1)

                # TODO: Compute entropy bonus to encourage exploration
                # Hint: entropy = dist.entropy().sum(dim=-1).mean()
                entropy = None  # TODO
                
                # TODO: Compute the probability ratio r(θ) = π_new(a|s) / π_old(a|s)
                # In log space: log r(θ) = log π_new - log π_old
                # Then: r(θ) = exp(log r(θ))
                # Hint: ratio = torch.exp(batch_log_probs - batch_old_lp)
                ratio = None  # TODO

                # TODO: Compute the clipped surrogate objective
                # surr1 = ratio * advantages (unclipped)
                # surr2 = clip(ratio, 1-ε, 1+ε) * advantages (clipped)
                # policy_loss = -min(surr1, surr2).mean()  (negative because we maximize)
                # Hint: Use torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = None  # TODO
                
                # TODO: Compute value function loss (MSE between predicted value and returns)
                # Hint: value_loss = 0.5 * ((batch_values.squeeze(-1) - batch_ret) ** 2).mean()
                value_loss = None  # TODO

                # TODO: Combine into total loss
                # loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
                # Note: entropy is SUBTRACTED because we want to MAXIMIZE entropy (more exploration)
                loss = None  # TODO

                # TODO: Backpropagation and optimizer step
                # 1. self.optimizer.zero_grad()
                # 2. loss.backward()
                # 3. Clip gradients: torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                # 4. self.optimizer.step()
                pass  # TODO

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy.item())

        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropy_losses)
        }
    
    def train(self, n_iterations, steps_per_iter=2048):
        """
        Main PPO training loop.
        
        The training loop follows this pattern:
        1. Collect trajectories (rollout phase)
        2. Compute advantages using GAE
        3. Update policy multiple times on collected data
        4. Repeat
        """
        # Initialize tracking lists
        training_stats = {
            'iterations': [],
            'mean_rewards': [],
            'policy_losses': [],
            'value_losses': [],
            'entropies': []
        }
        
        # Main training loop
        for iteration in range(n_iterations): 

            # TODO: Step 1 — Collect trajectories using current policy
            # Hint: trajectories = self.collect_trajectories(steps_per_iter)
            trajectories = None  # TODO

            # TODO: Step 2 — Compute GAE advantages and returns
            # Hint: advantages, returns = self.compute_gae(
            #           trajectories['rewards'], trajectories['values'], trajectories['dones'])
            advantages, returns = None, None  # TODO

            # TODO: Step 3 — Update the policy using collected data
            # Hint: update_stats = self.update_policy(
            #           trajectories['states'], trajectories['actions'],
            #           trajectories['log_probs'], advantages, returns)
            update_stats = None  # TODO

            # TODO: Step 4 — Log training statistics
            # Compute mean_reward = np.mean(trajectories['rewards'])
            # Append iteration, mean_reward, policy_loss, value_loss, entropy to training_stats
            pass  # TODO

            # Print training progress every 10 iterations
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration+1}/{n_iterations} | "
                      f"Mean Reward: {mean_reward:.2f} | "
                      f"Policy Loss: {update_stats['policy_loss']:.4f} | "
                      f"Value Loss: {update_stats['value_loss']:.4f} | "
                      f"Entropy: {update_stats['entropy']:.4f}")

        return training_stats

    def predict(self, state):
        """
        Get a deterministic action for deployment/evaluation (no sampling).
        
        Args:
            state: numpy array of shape (obs_dim,)
        
        Returns:
            action: numpy array of shape (action_dim,) in range (-1, 1)
        """
        # --- Deterministic tanh-squashed action (provided — not a TODO) ---
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            dist, _ = self.policy.forward(state_tensor)
            action = torch.tanh(dist.mean)  # deterministic, no sampling
        return action.cpu().numpy()[0]
