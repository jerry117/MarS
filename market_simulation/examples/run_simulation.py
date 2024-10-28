from market_simulation.agents.noise_agent import NoiseAgent
from matplotlib import dates
import matplotlib.pyplot as plt
from mlib.core.trade_info import TradeInfo
from typing import List
from pandas import Timestamp
from market_simulation.states.trade_info_state import TradeInfoState
from mlib.core.env import Env
from mlib.core.exchange import Exchange
from mlib.core.event import create_exchange_events
from mlib.core.exchange_config import create_exchange_config_without_call_auction
from pathlib import Path
import pandas as pd
import seaborn as sns


def run_simulation():
    """Run simulation with noise agent."""
    symbols = ["000000"]
    start_time = Timestamp("2024-01-01 09:30:00")
    end_time = Timestamp("2024-01-01 10:30:00")
    exchange_config = create_exchange_config_without_call_auction(
        market_open=start_time,
        market_close=end_time,
        symbols=symbols,
    )
    exchange = Exchange(exchange_config)
    agent = NoiseAgent(
        symbol=symbols[0],
        init_price=100000,
        interval_seconds=1,
        start_time=start_time,
        end_time=end_time,
    )
    exchange.register_state(TradeInfoState())
    env = Env(exchange=exchange, description="Noise agent simulation")
    env.register_agent(agent)
    env.push_events(create_exchange_events(exchange_config))
    for observation in env.env():
        action = observation.agent.get_action(observation)
        env.step(action)
    trade_infos: List[TradeInfo] = get_trade_infos(exchange, symbols[0], start_time, end_time)
    print(f"Get {len(trade_infos)} trade infos.")
    plot_price_curves(trade_infos, Path("tmp/price_curves.png"))


def get_trade_infos(exchange: Exchange, symbol: str, start_time: Timestamp, end_time: Timestamp):
    """Get trade infos from TradeInfoState."""
    state = exchange.states()[symbol][TradeInfoState.__name__]
    assert isinstance(state, TradeInfoState)
    trade_infos = state.trade_infos
    trade_infos = [x for x in trade_infos if start_time <= x.order.time <= end_time]
    return trade_infos


def plot_price_curves(trade_infos: List[TradeInfo], path: Path):
    """Plot price curves."""
    path.parent.mkdir(parents=True, exist_ok=True)
    prices = [
        {
            "Time": x.order.time,
            "Price": x.lob_snapshot.last_price,
        }
        for x in trade_infos
        if x.lob_snapshot.last_price > 0
    ]
    # group by 1 minute
    prices = pd.DataFrame(prices).groupby(pd.Grouper(key="Time", freq="1T")).mean().reset_index()
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.lineplot(x="Time", y="Price", data=prices, ax=ax)
    ax.xaxis.set_major_formatter(dates.DateFormatter("%H:%M"))
    ax.set_title("Price Trajectory Generated by NoiseAgent")
    fig.tight_layout()
    fig.savefig(str(path))
    plt.close(fig)
    print(f"Saved price curves to {path}")


if __name__ == "__main__":
    run_simulation()
