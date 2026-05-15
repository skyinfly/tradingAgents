"""盘中（实时）数据工具。

仅适用于 A 股市场。统一走 ``route_to_vendor`` 路由到 AKShare 实现。
agent 在收盘后调用这些工具会拿到上一交易日的累计快照，使用前请用
``main.resolve_trade_date`` 等机制判定当前是否处于盘中。
"""

from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_realtime_quote(
    symbol: Annotated[str, "A 股股票代码，如 002031"],
) -> str:
    """获取指定 A 股的实时报价与五档盘口。

    返回字段包括：最新价、涨跌幅、今开/最高/最低、成交量/额、换手率、
    量比、委比，以及买一到买五、卖一到卖五的价格与挂单量。
    盘中刷新延迟约 3-5 秒；非交易时段五档显示为 ``-``。
    """
    return route_to_vendor("get_realtime_quote", symbol)


@tool
def get_intraday_minute_bars(
    symbol: Annotated[str, "A 股股票代码"],
    period: Annotated[str, "分钟级别: 1/5/15/30/60"] = "5",
    lookback_minutes: Annotated[int, "回看分钟数（含今日盘中）"] = 240,
) -> str:
    """获取指定 A 股最近若干分钟的分钟 K 线（前复权，含今日盘中）。

    返回 CSV 文本，列含开/高/低/收/成交量/成交额/振幅/换手率等。
    适合做盘中量价配合、突破/放量判断；不要用它算 MACD/RSI 等
    依赖收盘价的日级指标。
    """
    return route_to_vendor("get_intraday_minute_bars", symbol, period, lookback_minutes)


@tool
def get_today_fund_flow_rank(
    top_n: Annotated[int, "返回前 N 名"] = 30,
    direction: Annotated[str, "inflow(净流入) 或 outflow(净流出)"] = "inflow",
) -> str:
    """获取全市场 A 股今日实时主力资金流排名。

    用于盘中确认目标股票相对全市场的资金动向位置：
    - direction=inflow 返回主力净流入 Top N
    - direction=outflow 返回主力净流出 Top N
    """
    return route_to_vendor("get_today_fund_flow_rank", top_n, direction)
