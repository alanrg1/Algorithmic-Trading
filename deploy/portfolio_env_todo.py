"""
Portfolio Optimization Environment

Gym environment for portfolio optimization using Deep RL.
Inherits from AlphaPortfolio for all portfolio logic.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from src.alpha_portfolio import AlphaPortfolio


class PortfolioEnv(AlphaPortfolio, gym.Env):
    """
    Gym environment for portfolio optimization.
    
    Inherits portfolio management from AlphaPortfolio.
    Handles gym interface: action/observation spaces, step, reset, reward.
    
    Action Space:
        Box(n_total) - Raw values -> softmax -> portfolio weights
    
    Observation Space:
        - Price features per stock
        - Current portfolio weights
        - Rolling covariance (flattened)
        - Rolling mean returns
        - Portfolio value (normalized)
        - Time step (normalized)
    
    Reward:
        Configurable: 'sharpe', 'sortino', 'return', 'utility'
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        stock_data: Dict[str, pd.DataFrame],
        initial_value: float = 100_000,
        transaction_cost_pct: float = 0.001,
        include_cash: bool = True,
        risk_free_rate: float = 0.0,
        reward_type: str = "sharpe",
        cov_window: int = 20,
    ):
        """
        Args:
            stock_data: Dict of ticker -> DataFrame with OHLCV + indicators
            initial_value: Starting portfolio value
            transaction_cost_pct: Transaction cost percentage
            include_cash: Whether to include cash as an asset
            risk_free_rate: Annual risk-free rate
            reward_type: 'sharpe', 'sortino', 'return', or 'utility'
            cov_window: Window for rolling statistics
        """
        # Store data
        self.stock_data = stock_data
        self.tickers = list(stock_data.keys())
        n_assets = len(self.tickers)
        
        # Initialize AlphaPortfolio
        AlphaPortfolio.__init__(
            self,
            n_assets=n_assets,
            initial_value=initial_value,
            transaction_cost_pct=transaction_cost_pct,
            include_cash=include_cash,
            risk_free_rate=risk_free_rate,
        )
        self._cov_window = cov_window
        
        # Environment config
        self.reward_type = reward_type
        
        # Calculate dimensions
        sample_df = next(iter(stock_data.values()))
        self.n_features = len(sample_df.columns)
        self.max_steps = min(len(df) for df in stock_data.values()) - 1
        
        # Precompute data
        self._precompute_data()
        
        # Observation dimensions
        n_cov = self.n_assets * (self.n_assets + 1) // 2
        self.obs_dim = (
            self.n_assets * self.n_features +  # Price features
            self.n_total +                      # Current weights
            n_cov +                             # Covariance (flattened)
            self.n_assets +                     # Mean returns
            2                                   # Value + step
        )
        
        # Action space: raw values -> softmax
        self.action_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(self.n_total,),
            dtype=np.float32
        )
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.obs_dim,),
            dtype=np.float32
        )
        
        # State
        self.current_step = 0
    
    def _precompute_data(self):
        """Precompute price matrix and returns."""
        min_len = min(len(df) for df in self.stock_data.values())
        
        # Price matrix (n_days, n_assets)
        self.price_matrix = np.zeros((min_len, self.n_assets))
        for i, ticker in enumerate(self.tickers):
            self.price_matrix[:, i] = self.stock_data[ticker]['Close'].values[:min_len]
        
        # Returns matrix
        self.returns_matrix = np.diff(np.log(self.price_matrix + 1e-8), axis=0)
        self.returns_matrix = np.nan_to_num(self.returns_matrix, nan=0.0)
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """Reset environment."""
        # Handle seeding for gymnasium
        if seed is not None:
            np.random.seed(seed)
        
        # Reset portfolio state
        AlphaPortfolio.reset(self)
        
        self.current_step = 0
        
        return self._get_obs(), self._get_info()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step.
        
        1. Convert action to weights
        2. Rebalance portfolio
        3. Advance time, update portfolio
        4. Compute reward
        """
        # =================================================================
        # TODO 1: Implement the step sequence
        # =================================================================
        # 1. Convert action to target weights
        target_weights = self.softmax(action)
        
        # 2. Get current prices and rebalance
        prices = self.price_matrix[self.current_step]
        self.rebalance(target_weights, prices)
        
        # 3. Advance time
        self.current_step += 1
        
        # 4. Check termination
        if self.current_step >= self.max_steps:
            return self._get_obs(), self._compute_reward(), True, False, self._get_info()
        
        # 5. Update portfolio with returns
        returns = self.returns_matrix[self.current_step]
        self.update(returns)
        
        # 6. Compute reward and check bankruptcy
        reward = None  # TODO
        terminated = None  # TODO
        
        return self._get_obs(), reward, terminated, False, self._get_info()
    
    def _get_obs(self) -> np.ndarray:
        """Construct observation vector."""
        obs_parts = []
        
        # 1. Price features
        for ticker in self.tickers:
            df = self.stock_data[ticker]
            idx = min(self.current_step, len(df) - 1)
            features = df.iloc[idx].values.astype(np.float32)
            features = np.clip(features / (np.abs(features) + 1e-8), -10, 10)
            obs_parts.append(features)
        
        # 2. Current weights
        obs_parts.append(self._weights.astype(np.float32))
        
        # 3. Rolling covariance (flattened)
        cov_flat = self.flatten_covariance()
        cov_flat = np.clip(cov_flat * 100, -10, 10)
        obs_parts.append(cov_flat.astype(np.float32))
        
        # 4. Rolling mean returns
        mean_ret = self.rolling_mean_returns()
        mean_ret = np.clip(mean_ret * 100, -10, 10)
        obs_parts.append(mean_ret.astype(np.float32))
        
        # 5. Normalized value
        value_norm = np.clip(self._value / self.initial_value, -10, 10)
        obs_parts.append(np.array([value_norm], dtype=np.float32))
        
        # 6. Normalized step
        step_norm = self.current_step / max(self.max_steps, 1)
        obs_parts.append(np.array([step_norm], dtype=np.float32))
        
        obs = np.concatenate(obs_parts)
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return obs.astype(np.float32)
    
    def _compute_reward(self) -> float:
        """Compute reward based on reward_type."""
        if len(self._returns_history) < 2:
            return 0.0
        
        # =================================================================
        # TODO 2: Implement reward computation
        # Use self.reward_type to select the branch
        # =================================================================
        if self.reward_type == "sharpe":
            reward = None  # TODO
        elif self.reward_type == "return":
            reward = None  # TODO
        else:
            reward = self._returns_history[-1] * 100
        
        return float(np.clip(reward, -10.0, 10.0))
    
    def _get_info(self) -> dict:
        """Get info dict."""
        info = {
            "portfolio_value": self._value,
            "weights": self._weights.copy(),
            "step": self.current_step,
            "total_return": self.total_return(),
        }
        
        if len(self._returns_history) > 1:
            info["sharpe"] = self.sharpe_ratio()
            info["max_drawdown"] = self.max_drawdown()
        
        return info
    
    def render(self, mode: str = "human"):
        """Render current state."""
        print(self.summary())


class PortfolioEnvWithBaselines(PortfolioEnv):
    """
    Portfolio environment that tracks classical baselines for comparison.
    
    Tracks:
    - Equal weight
    - Min-variance
    - Buy and hold
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.baselines = {
            "equal_weight": None,
            "min_variance": None,
            "buy_hold": None,
        }
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        obs, info = super().reset(seed=seed, options=options)
        
        # Initialize baseline portfolios
        for name in self.baselines:
            self.baselines[name] = AlphaPortfolio(
                n_assets=self.n_assets,
                initial_value=self.initial_value,
                transaction_cost_pct=self.transaction_cost_pct,
                include_cash=self.include_cash,
            )
        
        # Set initial weights
        prices = self.price_matrix[0]
        
        # Equal weight
        self.baselines["equal_weight"].rebalance(
            self.baselines["equal_weight"].equal_weights(), prices
        )
        
        # Buy and hold (equal weight, no rebalancing)
        self.baselines["buy_hold"].rebalance(
            self.baselines["buy_hold"].equal_weights(), prices
        )
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Update baselines
        if self.current_step > 0 and self.current_step < len(self.returns_matrix):
            returns = self.returns_matrix[self.current_step]
            prices = self.price_matrix[self.current_step]
            
            for name, portfolio in self.baselines.items():
                # Update with returns
                portfolio.update(returns)
                
                # Rebalance min-variance periodically
                if name == "min_variance" and self.current_step % 20 == 0:
                    portfolio.rebalance(portfolio.min_variance_weights(), prices)
        
        # Add baseline values to info
        for name, portfolio in self.baselines.items():
            info[f"baseline_{name}"] = portfolio.value
        
        return obs, reward, terminated, truncated, info
    
    def get_comparison(self) -> dict:
        """Get comparison of RL agent vs baselines."""
        comparison = {
            "RL Agent": {
                "value": self._value,
                "return": self.total_return(),
                "sharpe": self.sharpe_ratio(),
            }
        }
        
        for name, portfolio in self.baselines.items():
            if portfolio is not None:
                comparison[name] = {
                    "value": portfolio.value,
                    "return": portfolio.total_return(),
                    "sharpe": portfolio.sharpe_ratio(),
                }
        
        return comparison
