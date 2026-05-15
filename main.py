"""TradingAgents 单只 A 股分析示例。

特性：
1. 默认仅支持 A 股（数据源全部走 AKShare）
2. 自动判定当前 A 股市场状态并选取合适的 trade_date：
   - 已收盘 / 非交易日 → 使用最近的完整交易日
   - 盘中（含午休、下午盘）→ 使用今日，AKShare 会返回截至当下的盘中数据
   - 盘前 → 使用上一交易日（避免空数据）
3. 跑完后按"分析师 / 智能体"维度分段打印中文标签的最终产出
"""

import datetime as _dt

import pandas as pd
from dotenv import load_dotenv

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

load_dotenv()


# ==================== A 股交易时段与日期工具 ====================
_TRADING_DAYS_CACHE = None  # 通过 AKShare 拉到的全量交易日集合（成功后填充）


def _load_trading_days():
    """从 AKShare 拉取沪深交易日历（含节假日），失败时退化为周末判定。"""
    global _TRADING_DAYS_CACHE
    if _TRADING_DAYS_CACHE is not None:
        return _TRADING_DAYS_CACHE
    try:
        # 触发 sector tools 里的 requests UA 补丁，避免某些 AKShare 接口被服务端拒
        from tools.sector_data.akshare_sector import _patch_requests_default_ua
        _patch_requests_default_ua()
        import akshare as ak
        df = ak.tool_trade_date_hist_sina()
        _TRADING_DAYS_CACHE = set(pd.to_datetime(df["trade_date"]).dt.date)
    except Exception as e:
        print(f"[警告] 无法加载完整交易日历（{e}），回退至周末判定（节假日可能不准）")
        _TRADING_DAYS_CACHE = set()
    return _TRADING_DAYS_CACHE


def _is_trading_day(d: _dt.date) -> bool:
    days = _load_trading_days()
    if days:
        return d in days
    return d.weekday() < 5  # 0-4 是周一到周五


def _previous_trading_day(d: _dt.date) -> _dt.date:
    cur = d - _dt.timedelta(days=1)
    while not _is_trading_day(cur):
        cur -= _dt.timedelta(days=1)
    return cur


def resolve_trade_date(now: _dt.datetime = None) -> dict:
    """根据当前时刻智能选取分析基准日。

    返回包含 trade_date / status / note 的 dict，status 取值见函数体注释。
    """
    now = now or _dt.datetime.now()
    today = now.date()
    t = now.time()

    if not _is_trading_day(today):
        prev = _previous_trading_day(today)
        return {
            "trade_date": prev.strftime("%Y-%m-%d"),
            "status": "non_trading",
            "note": f"今天 {today} 非交易日，基准日选用最近交易日 {prev}",
        }

    market_open = _dt.time(9, 30)
    morning_close = _dt.time(11, 30)
    afternoon_open = _dt.time(13, 0)
    market_close = _dt.time(15, 0)

    if t >= market_close:
        return {
            "trade_date": today.strftime("%Y-%m-%d"),
            "status": "closed",
            "note": f"今天 {today} 已收盘（{t.strftime('%H:%M')}），使用当日完整数据进行分析",
        }
    if t >= afternoon_open:
        return {
            "trade_date": today.strftime("%Y-%m-%d"),
            "status": "intraday_pm",
            "note": (
                f"今天 {today} 下午盘中（{t.strftime('%H:%M')}），"
                f"将结合当日盘中数据（截至当下）与历史 K 线分析；"
                f"前一交易日为 {_previous_trading_day(today)}"
            ),
        }
    if t >= morning_close:
        return {
            "trade_date": today.strftime("%Y-%m-%d"),
            "status": "lunch_break",
            "note": (
                f"今天 {today} 午间休市（{t.strftime('%H:%M')}），"
                f"上午盘中数据已可用；前一交易日为 {_previous_trading_day(today)}"
            ),
        }
    if t >= market_open:
        return {
            "trade_date": today.strftime("%Y-%m-%d"),
            "status": "intraday_am",
            "note": (
                f"今天 {today} 上午盘中（{t.strftime('%H:%M')}），"
                f"将结合当日盘中数据（截至当下）与历史 K 线分析；"
                f"前一交易日为 {_previous_trading_day(today)}"
            ),
        }
    prev = _previous_trading_day(today)
    return {
        "trade_date": prev.strftime("%Y-%m-%d"),
        "status": "pre_market",
        "note": f"今天 {today} 尚未开盘（{t.strftime('%H:%M')}），基准日使用上一交易日 {prev}",
    }


# ==================== 自定义配置（A 股 + AKShare） ====================
config = DEFAULT_CONFIG.copy()
config["deep_think_llm"] = "gpt-5.4-mini"          # 复杂推理（研究经理、投组经理）
config["quick_think_llm"] = "gpt-5.4-mini"         # 快速分析（各分析师、辩手）
config["max_debate_rounds"] = 1                    # 多空辩论轮数

# A 股全部使用 AKShare
config["data_vendors"] = {
    "core_stock_apis": "akshare",
    "technical_indicators": "akshare",
    "fundamental_data": "akshare",
    "news_data": "akshare",
    "sector_data": "akshare",
}


# ==================== 运行图 ====================
TICKER = "002031"  # 巨轮智能

market = resolve_trade_date()
TRADE_DATE = market["trade_date"]

# 仅在 A 股交易时段（含午休）启用盘中分析师；盘前 / 收盘后 / 非交易日不启用
INTRADAY_STATUSES = {"intraday_am", "lunch_break", "intraday_pm"}
selected_analysts = ["market", "social", "news", "fundamentals"]
if market["status"] in INTRADAY_STATUSES:
    selected_analysts.append("intraday")

print("=" * 70)
print("  TradingAgents 单只 A 股分析")
print("=" * 70)
print(f"  股票代码  : {TICKER}")
print(f"  分析日期  : {TRADE_DATE}  [{market['status']}]")
print(f"  备注      : {market['note']}")
print(f"  启用分析师: {', '.join(selected_analysts)}")
print("=" * 70)

ta = TradingAgentsGraph(selected_analysts=selected_analysts, debug=True, config=config)
final_state, decision = ta.propagate(TICKER, TRADE_DATE)


# ==================== 各分析师/智能体最终产出汇总 ====================
def _print_section(label: str, content):
    """统一格式的分段打印。"""
    print(f"\n────────── 【{label}】 ──────────")
    if not content:
        print("(无内容)")
        return
    print(str(content).strip())


print("\n" + "=" * 70)
print("  各分析师 / 智能体最终产出汇总")
print("=" * 70)

# 第一阶段：分析师团队的报告
_print_section("市场分析师 · 技术面报告", final_state.get("market_report"))
_print_section("社交媒体分析师 · 舆情报告", final_state.get("sentiment_report"))
_print_section("新闻分析师 · 新闻报告", final_state.get("news_report"))
_print_section("基本面分析师 · 基本面报告", final_state.get("fundamentals_report"))
if "intraday" in selected_analysts:
    _print_section("盘中分析师 · 实时盘中态势", final_state.get("intraday_report"))

# 第二阶段：多空研究员辩论
inv_debate = final_state.get("investment_debate_state", {}) or {}
_print_section("多头研究员 · 看多论据", inv_debate.get("bull_history"))
_print_section("空头研究员 · 看空论据", inv_debate.get("bear_history"))
_print_section("研究经理 · 投资计划", final_state.get("investment_plan"))

# 第三阶段：交易员
_print_section("交易员 · 交易方案", final_state.get("trader_investment_plan"))

# 第四阶段：三方风险辩论 + 投资组合经理裁决
risk_debate = final_state.get("risk_debate_state", {}) or {}
_print_section("激进风险分析师 · 论点", risk_debate.get("aggressive_history"))
_print_section("保守风险分析师 · 论点", risk_debate.get("conservative_history"))
_print_section("中性风险分析师 · 论点", risk_debate.get("neutral_history"))
_print_section("投资组合经理 · 最终决策", final_state.get("final_trade_decision"))

# 一句话最终信号
print("\n" + "=" * 70)
print(f"  分析日期: {TRADE_DATE}  最终交易信号: {decision}")
print("=" * 70)

# 若需基于实际收益反思并更新记忆，取消下一行注释（参数为该笔交易的盈亏）
# ta.reflect_and_remember(1000)
