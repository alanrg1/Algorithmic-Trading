"""
LEARNING TODO: Live trading entry point (alpaca_live_todo.py)

This is the main script that ties everything together:
  (1) IEX WebSocket streams live minute bars in a background thread.
  (2) Every N minutes, we read the aggregated 15-min data, build an
      observation, run the policy, and place orders via REST.

The two TODOs here are the core trading loop and the single-trade cycle.
Everything else (model loading, REST wrappers, risk checks, setup) is
provided so you can focus on the orchestration logic.

Reference: alpaca_live.py (full solution)
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

from agent.actor_critic import ActorCritic
from alpaca_utils_todo import (
    TRAINING_TICKERS,
    prepare_features,
    build_observation,
    place_orders_from_actions,
    get_current_positions,
)
from alpaca_websocket import IEXStream15MinFetcher

load_dotenv(Path(__file__).parent / '.env')

RISK_PARAMS = {
    'max_position_size': 0.15,   # Max 15% per stock
    'daily_loss_limit': 0.05,    # Stop if down 5% in a day
    'max_drawdown': 0.30,        # Stop if down 30% from peak
    'min_trade_value': 100,      # Minimum $100 per trade
}
DEFAULT_MODEL_PATH = str(Path(__file__).parent.parent / 'models' / 'ppo_trading.pt')


class AlpacaPPOTrader:
    """Live trading manager."""

    def __init__(self, api, model_path: str, api_key: str = None, secret_key: str = None):
        self.api = api
        self._iex_fetcher = IEXStream15MinFetcher(api_key, secret_key)
        self.running = False

        # Performance tracking
        self.start_time = datetime.now()
        self.initial_value = None
        self.peak_value = None
        self.daily_start_value = None
        self.portfolio_history = []
        self.trade_step = 0

        self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load trained PPO policy from a .pt checkpoint."""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        obs_dim = checkpoint['obs_shape']
        act_dim = checkpoint['action_shape']

        self.policy = ActorCritic(obs_dim, act_dim)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy.eval()

        print(f"Policy loaded (obs={obs_dim}, act={act_dim})\n")

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Get deterministic action from the policy (mean of the distribution, squashed by tanh)."""
        state = torch.FloatTensor(observation).unsqueeze(0)
        with torch.no_grad():
            dist, _ = self.policy.forward(state)
            action = torch.tanh(dist.mean)
        return action.cpu().numpy()[0]

    def get_portfolio_value(self):
        """Get total portfolio value from Alpaca."""
        try:
            return float(self.api.get_account().portfolio_value)
        except Exception as e:
            print(f"Error getting portfolio value: {e}")
            return None

    def is_market_open(self):
        """Check if the US equity market is currently open."""
        try:
            return self.api.get_clock().is_open
        except Exception as e:
            print(f"  Error checking market status: {e}")
            return False

    def get_market_hours(self):
        """Get next market open and close times."""
        try:
            clock = self.api.get_clock()
            return clock.next_open, clock.next_close
        except Exception:
            return None, None

    def check_risk_limits(self):
        """Return False if a risk limit is breached (daily loss or max drawdown)."""
        value = self.get_portfolio_value()
        if value is None:
            return True

        if self.initial_value is None:
            self.initial_value = value
            self.peak_value = value
            self.daily_start_value = value

        if value > self.peak_value:
            self.peak_value = value

        daily_loss = (self.daily_start_value - value) / self.daily_start_value
        if daily_loss > RISK_PARAMS['daily_loss_limit']:
            print(f"\n RISK LIMIT: Daily loss {daily_loss*100:.2f}% > {RISK_PARAMS['daily_loss_limit']*100:.1f}%")
            return False

        drawdown = (self.peak_value - value) / self.peak_value
        if drawdown > RISK_PARAMS['max_drawdown']:
            print(f"\n RISK LIMIT: Drawdown {drawdown*100:.2f}% > {RISK_PARAMS['max_drawdown']*100:.1f}%")
            return False

        return True

    def fetch_latest_data(self):
        """Read latest 15-min bars from the WebSocket fetcher, then add features."""
        raw = self._iex_fetcher.get_latest_15min_data()
        if raw is None:
            print("No 15-min data from IEX stream yet.")
            return None
        if len(raw) < len(TRAINING_TICKERS):
            print(f"Only got {len(raw)}/{len(TRAINING_TICKERS)} tickers")
            return None
        return prepare_features(raw)

    # ==================================================================
    # TODO 1 — One trading cycle
    # ==================================================================
    def execute_trade(self):
        """
        TODO: Run one complete trading cycle.

        Think of it as a pipeline:

            safety check → data → observation → action → orders

        WHAT TO THINK ABOUT:

        1. GUARD RAILS FIRST — Before doing anything, check whether we're
           still within risk limits. If not, signal the caller to stop
           (return False). Also, you need the current portfolio value for
           later steps — what should you do if you can't read it?

        2. GET THE DATA — Call fetch_latest_data(). The WebSocket has been
           collecting bars in the background. If the data isn't ready yet
           (None), this isn't a fatal error — just skip this cycle.

        3. BUILD THE OBSERVATION — The policy needs the observation vector.
           You already implemented build_observation() in alpaca_utils_todo.
           Think about what arguments it needs: market features are in
           stock_data, but where do balance, positions, net_worth come from?
           (Hint: the REST API knows your account state.)

        4. ACT — Feed the observation into self.predict() to get the action
           array, then pass it to place_orders_from_actions().

        5. BOOKKEEPING — Record the portfolio value with a timestamp and
           increment the trade step counter.

        Returns: True to keep trading, False to stop (risk limit hit).
        """
        # TODO: implement
        raise NotImplementedError("execute_trade")

    # ==================================================================
    # TODO 2 — The main trading loop
    # ==================================================================
    def run(self, trade_frequency_minutes: int):
        """
        TODO: The main loop that keeps the trader alive.

        This function should run indefinitely (until interrupted or a risk
        limit is hit). 

        SETUP:
          - Start the WebSocket fetcher so bars begin streaming in the
            background BEFORE you enter the loop.
          - Initialize bookkeeping (running flag, last-trade timestamp, etc.)

        LOOP (while running):
          There are three things to decide on each iteration:

          a) NEW DAY? — If the calendar date changed since last check,
             reset the daily starting value (for the daily-loss risk check).

          b) MARKET OPEN? — There's no point trading when the exchange is
             closed. If it's closed, log when it reopens and sleep for a
             while. How long should you sleep? Too short wastes CPU, too
             long might miss the open.

          c) TIME TO TRADE? — You only want to trade every
             trade_frequency_minutes. Compare how many seconds have passed
             since the last trade. If enough time has passed, call
             execute_trade(). If it returns False, break out of the loop.
             Otherwise, sleep briefly and check again.

        CLEANUP:
          No matter how the loop ends (normal exit, Ctrl-C, exception),
          you must stop the WebSocket fetcher and print a summary.
          Think about which Python construct guarantees cleanup runs.

        HINTS:
          - time.sleep() for waiting, datetime.now() for timestamps.
          - float('inf') is a handy "first iteration" sentinel for elapsed.
          - Wrap the loop in try/except/finally for robust cleanup.
        """
        # TODO: implement
        raise NotImplementedError("run")

    def _print_summary(self):
        """Print final performance stats."""
        print(f"\nFINAL SUMMARY")
        fv = self.get_portfolio_value()
        if self.initial_value and fv:
            ret = (fv - self.initial_value) / self.initial_value * 100
            print(f"  Initial : ${self.initial_value:,.2f}")
            print(f"  Final   : ${fv:,.2f}")
            print(f"  Return  : {ret:+.2f}%")
            if self.peak_value:
                dd = (self.peak_value - fv) / self.peak_value * 100
                print(f"  Max DD  : {dd:.2f}%")
        print(f"  Runtime : {datetime.now() - self.start_time}")
        print(f"{'='*70}\n")


TRADE_FREQUENCY_MINUTES = 15


def main():
    """Load .env, connect to Alpaca REST, then run the trader."""
    api_key = os.getenv('ALPACA_API_KEY')
    secret = os.getenv('ALPACA_SECRET_KEY')
    base = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

    if not api_key or not secret:
        print("  Error: Alpaca API keys not found!")
        print("  Create deploy/.env with ALPACA_API_KEY and ALPACA_SECRET_KEY")
        return

    api = tradeapi.REST(api_key, secret, base, api_version='v2')

    try:
        acct = api.get_account()
        print(f"Connected — status: {acct.status}")
        print(f"Portfolio : ${float(acct.portfolio_value):,.2f}")
        print(f"Buying pwr: ${float(acct.buying_power):,.2f}")
    except Exception as e:
        print(f"  Connection failed: {e}")
        return

    trader = AlpacaPPOTrader(
        api, DEFAULT_MODEL_PATH,
        api_key=api_key,
        secret_key=secret,
    )
    trader.run(trade_frequency_minutes=TRADE_FREQUENCY_MINUTES)


if __name__ == '__main__':
    main()
