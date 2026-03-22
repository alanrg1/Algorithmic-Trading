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
                      balance: float, shares_held: Dict[str, int],
                      net_worth: float, max_net_worth: float,
                      current_step: int, max_steps: int,
                      initial_balance: float = 100000,
                      feat_mean: np.ndarray = None,
                      feat_std: np.ndarray = None) -> np.ndarray:
    """
    Build the observation vector that the PPO policy expects.

    The policy was trained on a fixed-size vector: for each ticker we pass
    its latest feature row (price, volume, indicators), then we append
    balance, shares held per ticker, net worth, max net worth, and a
    normalized step index. All values are normalized and clipped so the
    network never sees huge or invalid numbers.
    """
    obs = []

    for ticker in TRAINING_TICKERS:
        df = stock_data[ticker]
        row = df.iloc[-1]
        values = row.values.astype(np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6)
        if feat_mean is not None and feat_std is not None:
            values = (values - feat_mean) / feat_std
        obs.extend(values)

    # The training env (naive_env._next_observation) placed balance at
    # frame[-4 - n_tickers] = frame[239], which lands inside the stock
    # features block and overwrites the last ticker's last feature.
    # Replicate that layout exactly so obs dim == 263 (what the model expects).
    obs[239] = np.clip(balance / initial_balance, -10.0, 10.0)

    # Shares held (normalized)
    for ticker in TRAINING_TICKERS:
        obs.append(np.clip(shares_held.get(ticker, 0) / 1000.0, -10.0, 10.0))

    # Net worth (normalized)
    obs.append(np.clip(net_worth / initial_balance, -10.0, 10.0))

    # Max net worth (normalized)
    obs.append(np.clip(max_net_worth / initial_balance, -10.0, 10.0))

    # Step (normalized)
    obs.append(np.clip(current_step / max(max_steps, 1), 0.0, 1.0))

    obs = np.array(obs, dtype=np.float32)
    obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
    return np.clip(obs, -10.0, 10.0)


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


def place_orders_from_actions(api, actions: np.ndarray, tickers: List[str],
                              current_positions: Dict[str, int],
                              buying_power: float,
                              portfolio_value: float,
                              max_position_size: float = 1.0,
                              min_trade_value: float = 100) -> List[Dict]:
    """
    Turn policy actions into Alpaca orders, mirroring naive_env's step() logic.

    Buy  (action > 0): shares = int(remaining_buying_power * action / price).
                       Capped so position doesn't exceed max_position_size of portfolio.
    Sell (action < 0): shares = int(shares_held * |action|).
                       Never short-sells; capped by what we actually own.
    """
    orders = []

    # Local balance tracker mirrors naive_env's self.balance: decrements per buy.
    remaining_buying_power = buying_power

    for i, ticker in enumerate(tickers):
        action = float(actions[i])

        try:
            quote = api.get_latest_quote(ticker, feed='iex')
            ask = float(quote.ask_price or 0)
            bid = float(quote.bid_price or 0)
            if ask <= 0 and bid <= 0:
                print(f"  No valid quote for {ticker}, skipping")
                continue
            if ask <= 0:
                ask = bid
            if bid <= 0:
                bid = ask
            price = (ask + bid) / 2
        except Exception as e:
            print(f"  Could not get quote for {ticker}: {e}")
            continue

        if action > 0:
            # Alpaca validates buy orders against the ask price
            amount_to_spend = remaining_buying_power * action
            shares_to_buy = int(amount_to_spend / ask)

            # Enforce max position size: cap so total position stays within limit
            current_shares = current_positions.get(ticker, 0)
            current_value = current_shares * ask
            max_value = portfolio_value * max_position_size
            room = max(max_value - current_value, 0)
            shares_to_buy = min(shares_to_buy, int(room / ask))

            if shares_to_buy <= 0 or shares_to_buy * ask < min_trade_value:
                continue
            try:
                order = api.submit_order(
                    symbol=ticker, qty=shares_to_buy,
                    side='buy', type='market', time_in_force='day'
                )
                print(f"BUY {shares_to_buy:>4} {ticker} @ ~${ask:.2f}")
                remaining_buying_power -= shares_to_buy * ask
                orders.append({
                    'ticker': ticker, 
                    'side': 'buy',
                    'shares': shares_to_buy, 
                    'price': price, 
                    'order_id': order.id
                })
            except Exception as e:
                print(f"Error placing order for {ticker}: {e}")
        elif action < 0:
            shares_owned = current_positions.get(ticker, 0)
            # Mirror training: sell a fraction of shares held, never short-sell
            shares_to_sell = min(int(shares_owned * abs(action)), shares_owned)
            if shares_to_sell <= 0 or shares_to_sell * price < min_trade_value:
                continue
            try:
                order = api.submit_order(
                    symbol=ticker, qty=shares_to_sell,
                    side='sell', type='market', time_in_force='day'
                )
                print(f"SELL {shares_to_sell:>4} {ticker} @ ${price:.2f}")
                # Note: unlike training (where self.balance += sale immediately),
                # we do NOT add sell proceeds to remaining_buying_power here.
                # Per Alpaca docs: "sell long orders do not replenish your
                # available buying power until they are executed [filled]."
                # In a rapid-fire loop, sells may not fill before the next buy
                # is validated, so the proceeds aren't available yet.
                # https://docs.alpaca.markets/docs/orders-at-alpaca
                orders.append({
                    'ticker': ticker, 
                    'side': 'sell',
                    'shares': shares_to_sell, 'price': price, 
                    'order_id': order.id
                })
            except Exception as e:
                print(f"  Error placing order for {ticker}: {e}")

    return orders


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
