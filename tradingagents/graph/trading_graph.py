# TradingAgents/graph/trading_graph.py

import os
from pathlib import Path
import json
from datetime import date
from typing import Dict, Any, Tuple, List, Optional

from langgraph.prebuilt import ToolNode

from tradingagents.llm_clients import create_llm_client

from tradingagents.agents import *
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.utils.memory import FinancialSituationMemory
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)
from tradingagents.dataflows.config import set_config

# 从 agent_utils 导入新的抽象工具方法
from tradingagents.agents.utils.agent_utils import (
    get_stock_data,
    get_indicators,
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
    get_news,
    get_insider_transactions,
    get_global_news,
    get_realtime_quote,
    get_intraday_minute_bars,
    get_today_fund_flow_rank,
)

from .conditional_logic import ConditionalLogic
from .setup import GraphSetup
from .propagation import Propagator
from .reflection import Reflector
from .signal_processing import SignalProcessor


# 智能体英文标识 → 中文展示名映射，统一 debug 日志输出
_AGENT_ROLE_CN = {
    "Market Analyst": "市场分析师",
    "Social Media Analyst": "社交媒体分析师",
    "Social Analyst": "社交媒体分析师",
    "News Analyst": "新闻分析师",
    "Fundamentals Analyst": "基本面分析师",
    "Intraday Analyst": "盘中分析师",
    "Bull Researcher": "多头研究员",
    "Bull Analyst": "多头研究员",
    "Bear Researcher": "空头研究员",
    "Bear Analyst": "空头研究员",
    "Research Manager": "研究经理",
    "Trader": "交易员",
    "Aggressive": "激进风险分析师",
    "Aggressive Analyst": "激进风险分析师",
    "Risky Analyst": "激进风险分析师",
    "Conservative": "保守风险分析师",
    "Conservative Analyst": "保守风险分析师",
    "Safe Analyst": "保守风险分析师",
    "Neutral": "中性风险分析师",
    "Neutral Analyst": "中性风险分析师",
    "Judge": "投资组合经理",
    "Portfolio Manager": "投资组合经理",
    "Unknown": "未知角色",
}


def to_cn_role(role: str) -> str:
    """把英文角色名翻译成中文展示名；未识别时原样返回。"""
    if not role:
        return "未知角色"
    return _AGENT_ROLE_CN.get(role, role)


class TradingAgentsGraph:
    """负责协调 TradingAgents 框架的主类。"""

    def __init__(
        self,
        selected_analysts=["market", "social", "news", "fundamentals"],
        debug=False,
        config: Dict[str, Any] = None,
        callbacks: Optional[List] = None,
    ):
        """初始化 TradingAgents 图及其组件。

        Args:
            selected_analysts: 要包含的分析师类型列表
            debug: 是否以调试模式运行
            config: 配置字典；如果为 None，则使用默认配置
            callbacks: 可选的回调处理器列表（例如用于跟踪 LLM/工具统计信息）
        """
        self.debug = debug
        self.config = config or DEFAULT_CONFIG
        self.callbacks = callbacks or []

        # 更新接口配置
        set_config(self.config)

        # 创建必要的目录
        os.makedirs(
            os.path.join(self.config["project_dir"], "dataflows/data_cache"),
            exist_ok=True,
        )

        # 使用各 provider 对应的思考配置初始化 LLM
        llm_kwargs = self._get_provider_kwargs()

        # 如果提供了回调，则添加到 kwargs 中（传递给 LLM 构造函数）
        if self.callbacks:
            llm_kwargs["callbacks"] = self.callbacks

        deep_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["deep_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )
        quick_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["quick_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )

        self.deep_thinking_llm = deep_client.get_llm()
        self.quick_thinking_llm = quick_client.get_llm()
        
        # 初始化记忆模块
        self.bull_memory = FinancialSituationMemory("bull_memory", self.config)
        self.bear_memory = FinancialSituationMemory("bear_memory", self.config)
        self.trader_memory = FinancialSituationMemory("trader_memory", self.config)
        self.invest_judge_memory = FinancialSituationMemory("invest_judge_memory", self.config)
        self.portfolio_manager_memory = FinancialSituationMemory("portfolio_manager_memory", self.config)

        # 创建工具节点
        self.tool_nodes = self._create_tool_nodes()

        # 初始化各组件
        self.conditional_logic = ConditionalLogic(
            max_debate_rounds=self.config["max_debate_rounds"],
            max_risk_discuss_rounds=self.config["max_risk_discuss_rounds"],
        )
        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.tool_nodes,
            self.bull_memory,
            self.bear_memory,
            self.trader_memory,
            self.invest_judge_memory,
            self.portfolio_manager_memory,
            self.conditional_logic,
        )

        self.propagator = Propagator()
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        # 状态跟踪
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}  # 日期到完整状态字典的映射

        # 构建图
        self.graph = self.graph_setup.setup_graph(selected_analysts)

    def _get_provider_kwargs(self) -> Dict[str, Any]:
        """获取用于创建 LLM 客户端的 provider 专属 kwargs。"""
        kwargs = {}
        provider = self.config.get("llm_provider", "").lower()

        if provider == "google":
            thinking_level = self.config.get("google_thinking_level")
            if thinking_level:
                kwargs["thinking_level"] = thinking_level

        elif provider == "openai":
            reasoning_effort = self.config.get("openai_reasoning_effort")
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort

        elif provider == "anthropic":
            effort = self.config.get("anthropic_effort")
            if effort:
                kwargs["effort"] = effort

        return kwargs

    def _create_tool_nodes(self) -> Dict[str, ToolNode]:
        """使用抽象方法为不同数据源创建工具节点。"""
        return {
            "market": ToolNode(
                [
                    # 核心股票数据工具
                    get_stock_data,
                    # 技术指标
                    get_indicators,
                ]
            ),
            "social": ToolNode(
                [
                    # 用于社交媒体分析的新闻工具
                    get_news,
                ]
            ),
            "news": ToolNode(
                [
                    # 新闻与内幕交易信息
                    get_news,
                    get_global_news,
                    get_insider_transactions,
                ]
            ),
            "fundamentals": ToolNode(
                [
                    # 基本面分析工具
                    get_fundamentals,
                    get_balance_sheet,
                    get_cashflow,
                    get_income_statement,
                ]
            ),
            "intraday": ToolNode(
                [
                    # 盘中实时数据工具（A 股专用）
                    get_realtime_quote,
                    get_intraday_minute_bars,
                    get_today_fund_flow_rank,
                ]
            ),
        }

    def propagate(self, company_name, trade_date):
        """在指定日期为某家公司运行 TradingAgents 图。"""

        self.ticker = company_name

        # 初始化状态
        init_agent_state = self.propagator.create_initial_state(
            company_name, trade_date
        )
        args = self.propagator.get_graph_args()

        if self.debug:
            # 调试模式：updates 流定位发言节点，values 流保留完整 state
            debug_args = dict(args)
            debug_args["stream_mode"] = ["updates", "values"]

            final_state = None
            last_print_key = None  # (节点名, 最后一条消息 id) 防同节点重复打印

            for mode, payload in self.graph.stream(init_agent_state, **debug_args):
                if mode == "values":
                    final_state = payload
                    continue

                # mode == "updates"，payload 形如 {节点名: state_delta}
                if not isinstance(payload, dict):
                    continue
                for node_name, delta in payload.items():
                    # 过滤辅助节点（消息清理、工具调用本体）
                    if node_name.startswith("Msg Clear") or node_name.startswith("tools_"):
                        continue
                    if not isinstance(delta, dict):
                        continue
                    msgs = delta.get("messages") or []
                    if not msgs:
                        continue
                    last_msg = msgs[-1]
                    msg_id = getattr(last_msg, "id", None) or id(last_msg)
                    key = (node_name, msg_id)
                    if key == last_print_key:
                        continue
                    last_print_key = key
                    sender_cn = to_cn_role(node_name)
                    print(f"\n========== 【{sender_cn}】输出 ==========")
                    last_msg.pretty_print()

            if final_state is None:
                # 极端情况下没有 values 帧，回落到 invoke
                final_state = self.graph.invoke(init_agent_state, **args)
        else:
            # 不带追踪信息的标准模式
            final_state = self.graph.invoke(init_agent_state, **args)

        # 保存当前状态，供后续反思使用
        self.curr_state = final_state

        # 记录状态
        self._log_state(trade_date, final_state)

        # 返回决策和处理后的信号
        return final_state, self.process_signal(final_state["final_trade_decision"])

    def _resolve_debug_sender(self, chunk) -> str:
        """为 debug 流式输出尽力识别当前角色标签（仍返回英文，外层做中文映射）。"""
        sender = chunk.get("sender")
        if sender:
            return sender

        risk_speaker = chunk.get("risk_debate_state", {}).get("latest_speaker")
        if risk_speaker:
            return risk_speaker

        msg = chunk["messages"][-1] if chunk.get("messages") else None
        if msg is not None:
            msg_name = getattr(msg, "name", None)
            if msg_name:
                return msg_name
            additional_kwargs = getattr(msg, "additional_kwargs", {}) or {}
            if isinstance(additional_kwargs, dict) and additional_kwargs.get("name"):
                return additional_kwargs["name"]

        msg_content = getattr(msg, "content", None) if msg is not None else None
        if isinstance(msg_content, str) and msg_content:
            content_to_role = [
                ("final_trade_decision", "Portfolio Manager"),
                ("trader_investment_plan", "Trader"),
                ("investment_plan", "Research Manager"),
                ("market_report", "Market Analyst"),
                ("sentiment_report", "Social Media Analyst"),
                ("news_report", "News Analyst"),
                ("fundamentals_report", "Fundamentals Analyst"),
            ]
            for field, role in content_to_role:
                if chunk.get(field) == msg_content:
                    return role

        return "Unknown"

    def _log_state(self, trade_date, final_state):
        """将最终状态记录到 JSON 文件中。"""
        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "market_report": final_state["market_report"],
            "sentiment_report": final_state["sentiment_report"],
            "news_report": final_state["news_report"],
            "fundamentals_report": final_state["fundamentals_report"],
            "investment_debate_state": {
                "bull_history": final_state["investment_debate_state"]["bull_history"],
                "bear_history": final_state["investment_debate_state"]["bear_history"],
                "history": final_state["investment_debate_state"]["history"],
                "current_response": final_state["investment_debate_state"][
                    "current_response"
                ],
                "judge_decision": final_state["investment_debate_state"][
                    "judge_decision"
                ],
            },
            "trader_investment_decision": final_state["trader_investment_plan"],
            "risk_debate_state": {
                "aggressive_history": final_state["risk_debate_state"]["aggressive_history"],
                "conservative_history": final_state["risk_debate_state"]["conservative_history"],
                "neutral_history": final_state["risk_debate_state"]["neutral_history"],
                "history": final_state["risk_debate_state"]["history"],
                "judge_decision": final_state["risk_debate_state"]["judge_decision"],
            },
            "investment_plan": final_state["investment_plan"],
            "final_trade_decision": final_state["final_trade_decision"],
        }

        # 保存到文件
        directory = Path(self.config["results_dir"]) / self.ticker / "TradingAgentsStrategy_logs"
        directory.mkdir(parents=True, exist_ok=True)

        log_path = directory / f"full_states_log_{trade_date}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.log_states_dict[str(trade_date)], f, indent=4)

    def reflect_and_remember(self, returns_losses):
        """根据收益或损失对决策进行反思，并更新记忆。"""
        self.reflector.reflect_bull_researcher(
            self.curr_state, returns_losses, self.bull_memory
        )
        self.reflector.reflect_bear_researcher(
            self.curr_state, returns_losses, self.bear_memory
        )
        self.reflector.reflect_trader(
            self.curr_state, returns_losses, self.trader_memory
        )
        self.reflector.reflect_invest_judge(
            self.curr_state, returns_losses, self.invest_judge_memory
        )
        self.reflector.reflect_portfolio_manager(
            self.curr_state, returns_losses, self.portfolio_manager_memory
        )

    def process_signal(self, full_signal):
        """处理信号并提取核心决策。"""
        return self.signal_processor.process_signal(full_signal)
