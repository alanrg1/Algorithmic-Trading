"""
This module bridges the trained PPO policy and the live Alpaca broker.
Two key ideas:

  1. The policy expects the EXACT same observation vector it was trained on.
     Look at naive_env.py's _get_observation() — the vector layout (what goes
     where, how things are normalized) must match here or the policy will
     output nonsense.

  2. The policy outputs a continuous action per ticker in [-1, 1].
     You need to turn that into a dollar amount and then into a share count,
     and submit the order through Alpaca's REST API.

Reference: alpaca_utils.py (full solution)
REST API docs: https://github.com/alpacahq/alpaca-trade-api-python
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

def prepare_features(stock_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Add technical indicators (RSI, Bollinger Bands, MACD, etc.) to raw OHLCV.

    The policy was trained on data that included these features, so we must
    compute them here from the same formulas (see src/data_loader.add_technical_indicators).
    """
    processed = {}
    for ticker in TRAINING_TICKERS:
        if ticker not in stock_data:
            raise ValueError(f"Missing training ticker: {ticker}")
        processed[ticker] = add_technical_indicators(stock_data[ticker])
    return processed


# ---------------------------------------------------------------------------
# Build the observation vector
# ---------------------------------------------------------------------------

def build_observation(stock_data: Dict[str, pd.DataFrame],
                      balance: float, shares_held: Dict[str, int],
                      net_worth: float, max_net_worth: float,
                      current_step: int, max_steps: int,
                      initial_balance: float = 100000) -> np.ndarray:
    """
    WHY THIS MATTERS:
    The policy is a neural network that was trained on vectors of a specific
    shape and layout. If you change the order, leave something out, or forget
    to normalize, the network receives garbage and outputs garbage.

    WHAT TO DO:
    Open envs/naive_env.py and find _get_observation(). Your job is to
    reproduce that same vector here using live data instead of historical
    data. Walk through _get_observation() and mirror each section:

      - Market features: for every ticker, grab the latest row of indicator
        data. Think about what "latest row" means for a DataFrame and how
        to safely handle NaN / inf values.
      - Account state: the env appends balance, per-ticker holdings, net
        worth, max net worth, and a progress indicator. Each one is
        normalized so the values stay in a reasonable range for the network.
      - Final cleanup: the entire vector should be float32 with no NaN/inf
        values and clipped so nothing is extreme.

    HINTS:
      - The order of tickers must follow TRAINING_TICKERS exactly.
      - Look at how the env normalizes each piece (division by what?).
      - np.clip and np.nan_to_num are your friends.

    Returns: np.ndarray of shape (obs_dim,) with dtype float32.
    """
    obs = []

    for ticker in TRAINING_TICKERS:
        latest_features = stock_data[ticker].iloc[-1].values
        obs.extend(latest_features)

    obs.append(balance/initial_balance)

    for ticker in TRAINING_TICKERS:
        obs.append(shares_held.get(ticker, 0) / 1000.0)

    obs.append(net_worth/initial_balance)
    obs.append(max_net_worth/initial_balance)
    obs.append(current_step/max(max_steps, 1))
    obs_array = np.array(obs, dtype=np.float32)
    obs_array = np.nan_to_num(obs_array)
    obs_array = np.clip(obs_array, -10, 10)
    return obs_array

# ---------------------------------------------------------------------------
# This one is just a REST call — kept as-is.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Turn policy actions into real orders
# ---------------------------------------------------------------------------

def place_orders_from_actions(api, actions: np.ndarray, tickers: List[str],
                              portfolio_value: float, current_positions: Dict[str, int],
                              min_trade_value: float = 100) -> List[Dict]:
    """
    WHY THIS MATTERS:
    The policy outputs one float per ticker in [-1, 1]. Positive means
    "I want to buy", negative means "I want to sell", and the magnitude
    says how aggressively. You need to translate that intent into a
    concrete number of shares and submit it through the broker.

    WHAT TO DO:
    Think about the problem in three steps for each ticker:

      1. BUDGETING — The policy treats all tickers equally. How would you
         split the total portfolio value so each ticker gets a fair share?
         (Look at how naive_env sizes its trades for consistency.)

      2. PRICING — You need the current price to convert a dollar amount
         into a number of shares. Alpaca's REST API can give you the
         latest trade price for a symbol (use the 'iex' feed).
         What if the API call fails for one ticker? Should you skip it
         or crash?

      3. ORDERING — Once you know: direction (buy/sell), share count, and
         the ticker, submit a market order through the REST API.
         Consider: what's the minimum trade size worth executing?
         Should you cap the sell side for safety? (The training env does.)
         Use api.submit_order() — check the docs for the required params
         (symbol, qty, side, type, time_in_force).

    HINTS:
      - actions[i] corresponds to tickers[i].
      - int() truncates toward zero, which is the safe direction for shares.
      - Collect a list of dicts describing each order placed (ticker, side,
        shares, price) and return it so the caller can log them.

    Returns: list of order dicts.
    """
    executed_orders = []
    max_spend_per_ticker = portfolio_value/len(tickers)

    for i, ticker in enumerate(tickers):
        action = float(actions[i])

        if abs(action) < 0.01:
            continue

        try:
            latest_trade = api.get_latest_trade(ticker, feed='iex')
            current_price = float(latest_trade.price)
        except Exception as e:
            print(f"Error fetching trade price for {ticker}: {e}")
            continue

        if current_price <= 0:
            continue

        if action > 0:
            spend_amount = max_spend_per_ticker * action

            if spend_amount < min_trade_value:
                continue

            shares_to_buy = int(spend_amount/current_price)

            if shares_to_buy > 0:
                try:
                    order = api.submit_order(symbol=ticker, qty=shares_to_buy, side='buy', type='market', time_in_force='day')
                    executed_orders.append({'ticker': ticker, 'side': 'buy', 'qty': shares_to_buy, 'price': current_price, 'order_id': order.id})
                except Exception as e:
                    print(f"Error placing buy order for {ticker}: {e}")

        elif action < 0:
            shares_held = current_positions.get(ticker, 0)

            if shares_held <= 0:
                continue

            shares_to_sell = int(shares_held * abs(action))

            if shares_to_sell > 0:
                sell_value = shares_to_sell * current_price

                if sell_value < min_trade_value:
                    continue

                try:
                    order = api.submit_order(symbol=ticker, qty=shares_to_sell, side='sell', type='market', time_in_force='day')
                    executed_orders.append({'ticker': ticker, 'side': 'sell', 'qty': shares_to_sell, 'price': current_price,'order_id': order.id})
                except Exception as e:
                    print(f"Error placing sell order for {ticker}: {e}")

    return executed_orders

# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def calculate_portfolio_metrics(portfolio_values: List[float],
                                 returns: List[float]) -> Dict[str, float]:
    """Compute summary stats (total return, Sharpe, Sortino, max drawdown, win rate)."""
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
    """Pretty-print performance metrics."""
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
