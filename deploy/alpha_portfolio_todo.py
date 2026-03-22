import numpy as np 
import pandas as pd 
from typing import Dict, Optional, List
from abc import ABC 

class AlphaPortfolio: 
    def __init__(
        self, 
        n_assets: int, 
        initial_value: float = 100_000, 
        transaction_cost_pct: float = 0.001, 
        include_cash: bool = True, 
        risk_free_rate: float = 0.0 
    ): 
        self.n_assets = n_assets 
        self.n_total = n_assets + int(include_cash) 
        self.initial_value = initial_value  
        self.transaction_cost_pct = transaction_cost_pct  
        self.include_cash = include_cash   
        self.risk_free_rate = risk_free_rate   

        # Portfolio state 
        self._value = initial_value 
        self._weights = self._initial_weights()  
        self._holdings = np.zeros(self.n_assets) 
        self._cash = initial_value if include_cash else 0.0  

        # History 
        self._returns_history: List[float] = []
        self._weights_history: List[np.ndarray] = []  
        self._value_history: List[float] = [initial_value] 

        # Rolling statistics 
        self._returns_buffer: List[np.ndarray] = [] 
        self._cov_window = 20 

    def _initial_weights(self) -> np.ndarray: 
        if self.include_cash:  
            weights = np.zeros(self.n_total) 
            weights[-1] = 1.0 
        else: 
            weights = np.ones(self.n_total) / self.n_total 
        return weights 

    @property 
    def value(self) -> float: 
        return self._value 

    @property 
    def weights(self) -> np.ndarray: 
        return self._weights.copy()
    
    @property
    def cash(self) -> float: 
        return self._cash 
    
    @property 
    def holdings(self) -> np.ndarray: 
        return self._holdings.copy() 
    
    @property 
    def returns_history(self) -> np.ndarray:
        return np.array(self._returns_history) 

    @property 
    def value_history(self) -> np.ndarray:
        return np.array(self._value_history) 
    
    def reset(self): 
        self._value = self.initial_value 
        self._weights = self._initial_weights() 
        self._holdings = np.zeros(self.n_assets) 
        self._cash = self.initial_value if self.include_cash else 0.0  
        self._returns_history = [] 
        self._weights_history = []  
        self._value_history = [self.initial_value] 
        self._returns_buffer = [] 
    
    def rebalance(self, target_weights: np.ndarray, prices: np.ndarray) -> float: 
        """
        Rebalance portfolio to target weights.
        
        Args:
            target_weights: Target allocation (should sum to 1)
            prices: Current asset prices (length = n_assets)
        
        Returns:
            Transaction cost incurred
        """
        target_weights = self._normalize_weights(target_weights) 
        turnover = np.sum(np.abs(target_weights - self._weights)) 
        transaction_cost = turnover * self.transaction_cost_pct * self._value 
        self._value -= transaction_cost  

        old_weights = self._weights.copy()  
        self._weights = target_weights  

        if self.include_cash:  
            stock_weights = target_weights[:-1]   
            self._cash = target_weights[-1] * self._value  
        else: 
            stock_weights = target_weights   
            self._cash = 0.0 
        
        stock_value = stock_weights * self._value  
        self._holdings = stock_value / (prices + 1e-8) 
        self._weights_history.append(old_weights) 

        return transaction_cost 

    def update(self, returns: np.ndarray) -> float: 
        """
        Update portfolio value based on asset returns.
        
        Args:
            returns: Asset returns for the period (length = n_assets)
        
        Returns:
            Portfolio return for the period
        """ 
        # =====================================================================
        # TODO 1: Compute portfolio return and update state
        # (Week 6 slide 4: portfolio return is the weighted sum)
        # =====================================================================
        if self.include_cash: 
            assest_returns = np.concatenate([returns, [0.0]]) 
        else: 
            assest_returns = returns 

        portfolio_return = None  # TODO

        self._value *= None  # TODO

        self._returns_history.append(portfolio_return)
        self._value_history.append(self._value)
        self._returns_buffer.append(assest_returns) 

        if len(self._returns_buffer) > self._cov_window * 2:
            self._returns_buffer = self._returns_buffer[-self._cov_window * 2 :]
        
        return portfolio_return
    
    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray: 
        """ Normalize weights to sum to 1. """ 
        weights = np.asarray(weights).flatten()  
        if len(weights) != self.n_total:  
            raise ValueError(f"Expected {self.n_total} weights, got {len(weights)}")  
        
        weights = np.maximum(weights, 0.0) 
        total = np.sum(weights)  
        if total > 1e-8:
            weights = weights / total  
        else: 
            weights = self._initial_weights()  
        
        return weights 
    

    @staticmethod 
    def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:  
        """
        Convert raw values to valid portfolio weights via softmax.
        
        Args:
            x: Raw action values
            temperature: Higher = more uniform distribution
        
        Returns:
            Valid weights summing to 1
        """ 
        # =====================================================================
        # TODO 2: Implement numerically stable softmax
        # Hint: softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
        # =====================================================================
        x = np.asarray(x).flatten() / temperature 
        x = x - np.max(x)  # numerical stability
        exp_x = None  # TODO
        return None  # TODO
    
    def total_return(self) -> float:
        """Total return since inception (percentage)."""
        return (self._value / self.initial_value - 1) * 100
    
    def mean_return(self, annualize: bool = False) -> float:
        """Mean portfolio return."""
        if len(self._returns_history) == 0:
            return 0.0
        mean = np.mean(self._returns_history)
        if annualize:
            mean *= 252
        return mean
    
    def std_return(self, annualize: bool = False) -> float:
        """Standard deviation of returns."""
        if len(self._returns_history) < 2:
            return 0.0
        std = np.std(self._returns_history)
        if annualize:
            std *= np.sqrt(252)
        return std
    
    def sharpe_ratio(self, annualize: bool = True) -> float:
        """
        Sharpe ratio.
        
        Args:
            annualize: Whether to annualize
        
        Returns:
            Sharpe ratio
        """
        # =====================================================================
        # TODO 3: Implement the Sharpe ratio
        # Formula (Week 6 slide 15):
        #   Sharpe = (Expected Return - Risk-Free Rate) / Standard Deviation
        # =====================================================================
        mean = self.mean_return(annualize=annualize)
        std = self.std_return(annualize=annualize)
        
        if std < 1e-8:
            return 0.0
        
        rf = None  # TODO
        return None  # TODO
    
    def sortino_ratio(self, target: float = 0.0) -> float:
        """
        Sortino ratio (penalizes only downside volatility).
        
        Args:
            target: Minimum acceptable return
        
        Returns:
            Sortino ratio
        """
        if len(self._returns_history) < 2:
            return 0.0
        
        returns = np.array(self._returns_history)
        excess = returns - target
        downside = excess[excess < 0]
        
        if len(downside) == 0:
            return np.mean(excess) * 100  # All positive
        
        downside_std = np.sqrt(np.mean(downside ** 2))
        
        if downside_std < 1e-8:
            return 0.0
        
        return np.mean(excess) / downside_std
    
    def max_drawdown(self) -> float:
        """
        Maximum drawdown (percentage).
        
        Returns:
            Max drawdown as positive percentage (e.g., 0.2 = 20%)
        """
        if len(self._value_history) < 2:
            return 0.0
        
        values = np.array(self._value_history)
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / (peak + 1e-8)
        
        return float(np.max(drawdown))
    
    def calmar_ratio(self) -> float:
        """Calmar ratio (annual return / max drawdown)."""
        mdd = self.max_drawdown()
        if mdd < 1e-8:
            return 0.0
        
        ann_return = self.mean_return(annualize=True)
        return ann_return / mdd
    
    def information_ratio(self, benchmark_returns: np.ndarray) -> float:
        """
        Information ratio vs benchmark.
        
        Args:
            benchmark_returns: Benchmark returns array
        
        Returns:
            Information ratio
        """
        if len(self._returns_history) == 0:
            return 0.0
        
        port_returns = np.array(self._returns_history)
        min_len = min(len(port_returns), len(benchmark_returns))
        
        excess = port_returns[:min_len] - benchmark_returns[:min_len]
        
        if np.std(excess) < 1e-8:
            return 0.0
        
        return np.mean(excess) / np.std(excess)
    
    def get_metrics(self) -> dict:
        """Get all performance metrics as a dictionary."""
        return {
            "total_return": self.total_return(),
            "mean_return": self.mean_return(annualize=True),
            "std_return": self.std_return(annualize=True),
            "sharpe_ratio": self.sharpe_ratio(),
            "sortino_ratio": self.sortino_ratio(),
            "max_drawdown": self.max_drawdown() * 100,
            "calmar_ratio": self.calmar_ratio(),
            "final_value": self._value,
        }
    
    # =========================================================================
    # Rolling Statistics
    # =========================================================================
    
    def rolling_covariance(self, window: Optional[int] = None) -> np.ndarray:
        """
        Compute rolling covariance matrix of asset returns (stocks only, excludes cash).
        
        Args:
            window: Lookback window (default: self._cov_window)
        
        Returns:
            (n_assets, n_assets) covariance matrix
        """
        window = window or self._cov_window
        
        # =====================================================================
        # TODO 4: Compute the rolling covariance matrix
        # Week 6 slides 6-7: Σ captures how assets move together.
        # Diagonal = variance, off-diagonal = covariance.
        # =====================================================================
        if len(self._returns_buffer) < 2:
            return np.eye(self.n_assets) * 0.01
        
        returns = np.array(self._returns_buffer[-window:])
        
        if len(returns) < 2:
            return np.eye(self.n_assets) * 0.01
        
        # Exclude cash column (cash has zero variance)
        if self.include_cash and returns.shape[1] > self.n_assets:
            returns = returns[:, :self.n_assets]
        
        cov = None  # TODO
        
        if cov.ndim == 0:
            cov = np.array([[cov]])
        
        return cov
    
    def rolling_mean_returns(self, window: Optional[int] = None) -> np.ndarray:
        """
        Compute rolling mean returns (stocks only, excludes cash).
        
        Args:
            window: Lookback window
        
        Returns:
            (n_assets,) mean returns
        """
        window = window or self._cov_window
        
        if len(self._returns_buffer) == 0:
            return np.zeros(self.n_assets)
        
        returns = np.array(self._returns_buffer[-window:])
        
        # Only use stock returns (exclude cash column if include_cash)
        if self.include_cash and returns.shape[1] > self.n_assets:
            returns = returns[:, :self.n_assets]
        
        return np.mean(returns, axis=0)
    
    # =========================================================================
    # Classical Optimization Strategies
    # =========================================================================
    
    def equal_weights(self) -> np.ndarray:
        """
        Equal weight portfolio (1/n).
        
        Returns:
            (n_total,) weights
        """
        if self.include_cash:
            # Equal weight stocks, no cash
            weights = np.zeros(self.n_total)
            weights[:-1] = 1.0 / self.n_assets
        else:
            weights = np.ones(self.n_total) / self.n_total
        
        return weights
    
    def min_variance_weights(self, cov_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Minimum variance portfolio.
        
        min w^T Σ w  subject to Σw = 1
        
        Args:
            cov_matrix: Covariance matrix (default: use rolling)
        
        Returns:
            (n_total,) optimal weights
        """
        if cov_matrix is None:
            cov_matrix = self.rolling_covariance()
        
        n = cov_matrix.shape[0]
        ones = np.ones(n)
        
        # =====================================================================
        # TODO 5: Compute minimum variance portfolio weights
        # Week 6 slide 11 — closed-form:
        #   w* = Σ^{-1} / (1^T Σ^{-1} 1)
        # =====================================================================
        try:
            cov_reg = cov_matrix + np.eye(n) * 1e-6
            inv_cov = np.linalg.inv(cov_reg)
            stock_weights = None  # TODO
            
            # Long-only constraint: clamp negatives to 0, re-normalize
            stock_weights = np.maximum(stock_weights, 0)
            stock_weights = stock_weights / (np.sum(stock_weights) + 1e-8)
            
        except np.linalg.LinAlgError:
            stock_weights = np.ones(n) / n
        
        # Add cash weight (0)
        if self.include_cash:
            weights = np.concatenate([stock_weights, [0.0]])
        else:
            weights = stock_weights
        
        return weights
    
    def max_sharpe_weights(
        self, 
        mean_returns: Optional[np.ndarray] = None,
        cov_matrix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Maximum Sharpe ratio portfolio.
        
        max (w^T μ - r_f) / sqrt(w^T Σ w)
        
        Args:
            mean_returns: Expected returns (default: use rolling)
            cov_matrix: Covariance matrix (default: use rolling)
        
        Returns:
            (n_total,) optimal weights
        """
        if mean_returns is None:
            mean_returns = self.rolling_mean_returns()
        if cov_matrix is None:
            cov_matrix = self.rolling_covariance()
        
        n = cov_matrix.shape[0]
        
        # =====================================================================
        # TODO 6: Compute maximum Sharpe (tangency) portfolio weights
        # Week 6 slides 19-20 — closed-form:
        #   w* = Σ^{-1} (μ - R_f) / (1^T Σ^{-1} (μ - R_f))
        # =====================================================================
        excess = mean_returns - self.risk_free_rate / 252
        
        try:
            cov_reg = cov_matrix + np.eye(n) * 1e-6
            inv_cov = np.linalg.inv(cov_reg)
            stock_weights = None  # TODO
            
            # Long-only constraint: clamp negatives to 0, re-normalize
            stock_weights = np.maximum(stock_weights, 0)
            stock_weights = stock_weights / (np.sum(stock_weights) + 1e-8)
            
        except np.linalg.LinAlgError:
            stock_weights = np.ones(n) / n
        
        if self.include_cash:
            weights = np.concatenate([stock_weights, [0.0]])
        else:
            weights = stock_weights
        
        return weights
    
    def risk_parity_weights(
        self, 
        cov_matrix: Optional[np.ndarray] = None,
        n_iter: int = 50
    ) -> np.ndarray:
        """
        Risk parity portfolio (equal risk contribution).
        
        Args:
            cov_matrix: Covariance matrix
            n_iter: Optimization iterations
        
        Returns:
            (n_total,) optimal weights
        """
        if cov_matrix is None:
            cov_matrix = self.rolling_covariance()
        
        n = cov_matrix.shape[0]
        stock_weights = np.ones(n) / n
        
        for _ in range(n_iter):
            port_vol = np.sqrt(stock_weights @ cov_matrix @ stock_weights + 1e-8)
            marginal = cov_matrix @ stock_weights / port_vol
            risk_contrib = stock_weights * marginal
            
            stock_weights = stock_weights / (risk_contrib + 1e-8)
            stock_weights = stock_weights / np.sum(stock_weights)
        
        if self.include_cash:
            weights = np.concatenate([stock_weights, [0.0]])
        else:
            weights = stock_weights
        
        return weights
    
    # =========================================================================
    # Utilities
    # =========================================================================
    
    def flatten_covariance(self, cov_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Flatten upper triangle of covariance matrix for observation space.
        
        Args:
            cov_matrix: Covariance matrix (default: use rolling)
        
        Returns:
            Flattened upper triangle
        """
        if cov_matrix is None:
            cov_matrix = self.rolling_covariance()
        
        indices = np.triu_indices(cov_matrix.shape[0])
        return cov_matrix[indices]
    
    def summary(self) -> str:
        """Get portfolio summary as string."""
        metrics = self.get_metrics()
        
        lines = [
            "=" * 50,
            "PORTFOLIO SUMMARY",
            "=" * 50,
            f"Value:        ${self._value:,.2f}",
            f"Total Return: {metrics['total_return']:+.2f}%",
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}",
            f"Max Drawdown: {metrics['max_drawdown']:.2f}%",
            "",
            "Weights:",
        ]
        
        for i, w in enumerate(self._weights):
            if self.include_cash and i == len(self._weights) - 1:
                lines.append(f"  Cash:     {w*100:.1f}%")
            else:
                lines.append(f"  Asset {i}: {w*100:.1f}%")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)
