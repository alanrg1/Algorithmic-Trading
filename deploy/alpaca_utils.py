"""
Alpaca trading helpers: feature prep, observation building, orders, positions.

Live market data is provided by the IEX WebSocket (see alpaca_websocket.py).
This module handles: turning raw OHLCV into features, building the observation
vector for the policy, and calling the Alpaca REST API to read account/positions
and place orders.

REST API Alpaca: https://github.com/alpacahq/alpaca-trade-api-python?tab=readme-ov-file
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import add_technical_indicators

# Ticker list must match the one used during training (order matters for the policy).
TRAINING_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
    'META', 'TSLA', 'NFLX', 'UNH', 'JNJ',
    'V', 'JPM', 'WMT', 'MA', 'PG',
    'HD', 'DIS', 'BAC', 'XOM', 'CVX'
]

# Bollinger Bands and RVOL each use a 20-period rolling window — the largest
# lookback in add_technical_indicators. After the dropna() that function calls,
# any DataFrame with fewer than 20 rows will be completely empty.
MIN_BARS_FOR_INDICATORS = 20

def softmax(x):
    """Compute softmax values for action array."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def prepare_features(stock_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Add technical indicators (RSI, Bollinger Bands, MACD, etc.) to raw OHLCV.

    The policy was trained on data that included these features, so we must
    compute them here from the same formulas (see src/data_loader.add_technical_indicators).
    """
    for ticker in TRAINING_TICKERS:
        if ticker not in stock_data:
            raise ValueError(f"Missing training ticker: {ticker}")
        n = len(stock_data[ticker])
        if n < MIN_BARS_FOR_INDICATORS:
            print(
                f"  Indicator warm-up: {ticker} has {n}/{MIN_BARS_FOR_INDICATORS} bars. "
                f"Waiting for more 15-min bars..."
            )
            return None

    processed = {}
    for ticker in TRAINING_TICKERS:
        processed[ticker] = add_technical_indicators(stock_data[ticker])
    return processed


def build_observation(stock_data: Dict[str, pd.DataFrame],
                      current_weights: np.ndarray,
                      net_worth: float,
                      initial_value: float,
                      current_step: int,
                      max_steps: int = 252) -> np.ndarray:
    """
    Build the 493-element observation vector expected by PortfolioEnv.
    """
    obs_parts = []

    # 1. Price features (Normalized exactly like the training env)
    for ticker in TRAINING_TICKERS:
        df = stock_data[ticker]
        features = df.iloc[-1].values.astype(np.float32)
        features = np.clip(features / (np.abs(features) + 1e-8), -10, 10)
        obs_parts.append(features)

    # Calculate historical returns for Covariance and Means (last 21 closes = 20 returns)
    returns_list = []
    for ticker in TRAINING_TICKERS:
        closes = stock_data[ticker]['Close'].values[-21:]
        rets = np.diff(np.log(closes + 1e-8))
        returns_list.append(rets)
    returns_matrix = np.column_stack(returns_list)

    # 2. Current portfolio weights
    obs_parts.append(current_weights.astype(np.float32))

    # 3. Rolling covariance (flattened upper triangle)
    if len(returns_matrix) > 1:
        cov_matrix = np.cov(returns_matrix, rowvar=False)
    else:
        cov_matrix = np.zeros((len(TRAINING_TICKERS), len(TRAINING_TICKERS)))

    cov_flat = cov_matrix[np.triu_indices_from(cov_matrix)]
    cov_flat = np.clip(cov_flat * 100, -10, 10)
    obs_parts.append(cov_flat.astype(np.float32))

    # 4. Rolling mean returns
    mean_ret = np.mean(returns_matrix, axis=0) if len(returns_matrix) > 0 else np.zeros(len(TRAINING_TICKERS))
    mean_ret = np.clip(mean_ret * 100, -10, 10)
    obs_parts.append(mean_ret.astype(np.float32))

    # 5. Normalized Net Worth
    value_norm = np.clip(net_worth / initial_value, -10, 10)
    obs_parts.append(np.array([value_norm], dtype=np.float32))

    # 6. Normalized Step
    step_norm = current_step / max(max_steps, 1)
    obs_parts.append(np.array([step_norm], dtype=np.float32))

    obs = np.concatenate(obs_parts)
    obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
    return obs.astype(np.float32)


def place_orders_from_actions(api, actions: np.ndarray, tickers: List[str],
                              current_positions: Dict[str, int],
                              stock_data: Dict[str, pd.DataFrame],
                              portfolio_value: float,
                              min_trade_value: float = 100) -> List[Dict]:
    """
    Target Weight Rebalancing Engine.
    Applies Softmax to actions, calculates target dollar amounts, and issues Sells then Buys.
    """
    orders = []

    # Convert AI raw logits to target portfolio percentages
    target_weights = softmax(actions)

    sells = []
    buys = []

    for i, ticker in enumerate(tickers):
        target_weight = target_weights[i]
        target_dollar_value = portfolio_value * target_weight

        # Use the latest 15-min bar close as our reference price
        price = float(stock_data[ticker]['Close'].iloc[-1])

        current_shares = current_positions.get(ticker, 0)
        current_dollar_value = current_shares * price

        # Find the dollar difference required to reach the target weight
        delta_dollars = target_dollar_value - current_dollar_value

        # Ignore trades smaller than min_trade_value to prevent micro-churning
        if abs(delta_dollars) < min_trade_value:
            continue

        shares_to_trade = int(abs(delta_dollars) / price)
        if shares_to_trade == 0:
            continue

        if delta_dollars < 0:
            # We are overweight, need to SELL
            shares_to_sell = min(shares_to_trade, current_shares)
            sells.append((ticker, shares_to_sell, price))
        else:
            # We are underweight, need to BUY
            buys.append((ticker, shares_to_trade, price))

    # Execute SELLS first to free up buying power
    for ticker, shares, price in sells:
        try:
            order = api.submit_order(
                symbol=ticker, qty=shares,
                side='sell', type='market', time_in_force='day'
            )
            print(f"SELL {shares:>4} {ticker} @ ~${price:.2f} (Target Rebalance)")
            orders.append({'ticker': ticker, 'side': 'sell', 'shares': shares})
        except Exception as e:
            print(f"  Error selling {ticker}: {e}")

    # Give Alpaca 1 second to process the sells and release buying power
    import time
    time.sleep(1)

    # Execute BUYS
    for ticker, shares, price in buys:
        try:
            order = api.submit_order(
                symbol=ticker, qty=shares,
                side='buy', type='market', time_in_force='day'
            )
            print(f"BUY  {shares:>4} {ticker} @ ~${price:.2f} (Target Rebalance)")
            orders.append({'ticker': ticker, 'side': 'buy', 'shares': shares})
        except Exception as e:
            print(f"  Error buying {ticker}: {e}")

    return orders

def get_current_positions(api) -> Dict[str, int]:
    """
    Call Alpaca REST API to get current holdings.

    Returns a dict {ticker: share_count}. Only tickers with a position
    are included; the trading logic treats missing tickers as 0 shares.
    """
    positions = {}
    try:
        for pos in api.list_positions():
            positions[pos.symbol] = int(pos.qty)
    except Exception as e:
        print(f"Error fetching positions: {e}")
    return positions

def calculate_portfolio_metrics(portfolio_values: List[float],
                                 returns: List[float]) -> Dict[str, float]:
    """Compute summary stats (total return, Sharpe, Sortino, max drawdown, win rate) from value and return series."""
    pv = np.array(portfolio_values)
    ret = np.array(returns)

    sharpe = (np.mean(ret) / (np.std(ret) + 1e-8) * np.sqrt(252)) if len(ret) > 1 else 0.0

    downside = ret[ret < 0]
    sortino = (np.mean(ret) / (np.std(downside) + 1e-8) * np.sqrt(252)) if len(downside) > 0 else sharpe

    peak = np.maximum.accumulate(pv)
    max_dd = float(np.max((peak - pv) / peak) * 100)

    total_return = ((pv[-1] - pv[0]) / pv[0]) * 100
    win_rate = float(np.mean(ret > 0) * 100) if len(ret) > 0 else 0.0

    return {
        'total_return': float(total_return),
        'sharpe_ratio': float(sharpe),
        'sortino_ratio': float(sortino),
        'max_drawdown': float(max_dd),
        'win_rate': float(win_rate),
        'final_value': float(pv[-1]),
        'num_periods': len(portfolio_values),
    }


def print_performance_summary(metrics: Dict[str, float], title: str = "Performance Summary"):
    """Print a table of performance metrics (return %, Sharpe, drawdown, etc.) to the console."""
    print(f"\n{'='*55}")
    print(f"{title:^55}")
    print(f"{'='*55}")
    print(f"Total Return:      {metrics['total_return']:>8.2f}%")
    print(f"Sharpe Ratio:      {metrics['sharpe_ratio']:>8.2f}")
    print(f"Sortino Ratio:     {metrics['sortino_ratio']:>8.2f}")
    print(f"Max Drawdown:      {metrics['max_drawdown']:>8.2f}%")
    print(f"Win Rate:          {metrics['win_rate']:>8.1f}%")
    print(f"Final Value:       ${metrics['final_value']:>,.2f}")
    print(f"Trading Days:      {metrics['num_periods']:>8,}")
    print(f"{'='*55}\n")
