"""盘中分析师（Intraday Analyst）。

仅在 A 股交易时段（含午休）启用，负责：
- 拉取实时报价 + 五档盘口
- 拉取最近 N 分钟的分钟 K 线（量价配合）
- 对照全市场今日资金流排名，判断目标股的相对动向
- 输出一段"盘中态势"报告写入 ``intraday_report``

刻意不依赖日线技术指标（MACD/RSI/布林带）—— 那些依赖收盘价，盘中失真。
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_intraday_minute_bars,
    get_language_instruction,
    get_realtime_quote,
    get_today_fund_flow_rank,
)


def create_intraday_analyst(llm):
    def intraday_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_realtime_quote,
            get_intraday_minute_bars,
            get_today_fund_flow_rank,
        ]

        system_message = (
            """You are the Intraday Analyst specialized in A-share short-term market microstructure.

You ONLY analyze intraday signals; do NOT analyze fundamentals, long-term technicals (50/200 SMA, MACD daily, etc.) or social media sentiment — other analysts cover those.

Tool usage rules:
1. ALWAYS call `get_realtime_quote(symbol)` first to anchor the latest price, change %, turnover, volume ratio (量比), and 5-level order book (买卖五档).
2. Then call `get_intraday_minute_bars(symbol, period='5', lookback_minutes=240)` to look at the most recent intraday 5-minute bars. Prefer 5-minute as a balance; switch to 1-minute only when you need to inspect a specific sharp move.
3. Finally call `get_today_fund_flow_rank(top_n=30)` once to see if the target stock is in the top inflow list of the whole market today, and contextualize its relative strength.
4. Do NOT call any of these tools more than twice in this session — they hit live endpoints with rate limits.

When writing the report, structure your findings as:

1. **盘中态势 (Intraday Snapshot)**: latest price + change % + open price + high/low; one sentence on whether price is closer to today's high or low.
2. **量价配合 (Volume-Price)**: 量比 + 换手率 + the trajectory of recent 30-60 minutes (rising/sideways/falling, with volume confirming or diverging).
3. **盘口结构 (Order Book)**: read the 5-level bid/ask — large buy-side stack ≠ bullish necessarily; large sell-side absorption matters. Note 委比 sign.
4. **资金流定位 (Fund Flow Positioning)**: is the stock in today's market-wide inflow top list? Cross-reference with sector behavior if obvious from the data.
5. **盘中倾向 (Intraday Lean)**: a 1-2 sentence call on short-term direction (next 30-120 minutes). Be explicit about uncertainty. If signals conflict, say so — do NOT force a directional call.

IMPORTANT constraints:
- The report must be tagged at the top with timestamp and current trade_date.
- Do NOT make BUY/HOLD/SELL recommendations — that's the Trader's and Portfolio Manager's job. You only describe what intraday data is saying.
- If `get_realtime_quote` shows order book as all dashes (`-`), state explicitly that market is closed and the report uses last-session data only.
- End with a Markdown table summarizing the 5-6 numerical anchors (price / change / 量比 / 换手率 / 主力净流入 / order book imbalance).
"""
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""
        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "intraday_report": report,
        }

    return intraday_analyst_node
