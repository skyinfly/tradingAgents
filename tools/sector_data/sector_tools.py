from io import StringIO
from typing import Annotated

import pandas as pd
from langchain_core.tools import tool

from tradingagents.dataflows.interface import route_to_vendor


def _tool_csv_to_df(text: str) -> pd.DataFrame:
    """把工具返回的带注释 CSV 文本解析成 DataFrame。"""
    if not isinstance(text, str):
        return pd.DataFrame()
    if text.startswith("Error ") or text.startswith("No "):
        return pd.DataFrame()

    lines = [line for line in text.splitlines() if line and not line.startswith("#")]
    if not lines:
        return pd.DataFrame()

    csv_text = "\n".join(lines)
    try:
        # 强制把代码/证券代码当字符串读，避免 000591 被推断成 591
        return pd.read_csv(StringIO(csv_text), dtype={"代码": str, "证券代码": str, "板块代码": str})
    except Exception:
        return pd.DataFrame()


def _pick_sector_name_column(df: pd.DataFrame) -> str | None:
    """从不同数据源字段名中识别板块或股票名称列。"""
    candidates = ["板块名称", "名称", "行业", "概念名称", "概念", "股票名称", "name", "Name"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _pick_code_column(df: pd.DataFrame) -> str | None:
    """识别原始代码列，用于保留数据源返回的代码字段。"""
    candidates = ["代码", "证券代码", "板块代码", "code", "Code"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _pick_ticker_column(df: pd.DataFrame) -> str | None:
    """识别可用于后续个股分析的股票代码列。"""
    candidates = ["代码", "证券代码", "symbol", "Symbol"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalize_indicator_input(indicator: str) -> str:
    """兼容命令行编码异常和英文简写。"""
    value = (indicator or "今日").strip()
    aliases = {
        "today": "今日",
        "day": "今日",
        "1d": "今日",
        "5d": "5日",
        "10d": "10日",
        "??": "今日",
        "5?": "5日",
        "10?": "10日",
    }
    return aliases.get(value.lower(), value)


@tool
def get_market_sectors(
    sector_type: Annotated[str, "industry or concept"] = "industry",
) -> str:
    """获取当前市场板块列表。

    Args:
        sector_type: 板块类型，支持 industry/行业 或 concept/概念。

    Returns:
        带元信息注释的 CSV 文本。
    """
    return route_to_vendor("get_market_sectors", sector_type)


@tool
def get_sector_fund_flow(
    indicator: Annotated[str, "今日/5日/10日"] = "今日",
    sector_type: Annotated[str, "industry or concept"] = "industry",
    top_n: Annotated[int, "number of sectors to return"] = 50,
) -> str:
    """获取板块资金流排行。

    Args:
        indicator: 资金流统计窗口，常用值为 今日、5日、10日。
        sector_type: 板块类型，支持 industry/行业 或 concept/概念。
        top_n: 返回排行前多少个板块。

    Returns:
        带元信息注释的 CSV 文本。
    """
    return route_to_vendor("get_sector_fund_flow", _normalize_indicator_input(indicator), sector_type, top_n)


@tool
def get_sector_constituents(
    sector_name: Annotated[str, "sector name, e.g. 小金属 / 融资融券"],
    sector_type: Annotated[str, "industry or concept"] = "industry",
    top_n: Annotated[int, "number of constituents to return"] = 200,
) -> str:
    """获取指定板块的成分股。

    Args:
        sector_name: 板块名称，例如 小金属、融资融券。
        sector_type: 板块类型，支持 industry/行业 或 concept/概念。
        top_n: 最多返回多少只成分股。

    Returns:
        带元信息注释的 CSV 文本。
    """
    return route_to_vendor("get_sector_constituents", sector_name, sector_type, top_n)


@tool
def build_sector_stock_candidates(
    indicator: Annotated[str, "fund flow window: 今日/5日/10日"] = "今日",
    sector_type: Annotated[str, "industry or concept"] = "industry",
    top_sectors: Annotated[int, "how many top sectors to include"] = 5,
    stocks_per_sector: Annotated[int, "how many stocks per sector"] = 20,
    include_today: Annotated[bool, "include today's main fund inflow columns"] = True,
    include_yesterday: Annotated[bool, "include previous trading day's main fund inflow columns"] = False,
) -> str:
    """根据板块资金流构建待分析股票候选池。

    这个工具只做数据准备：先取资金流 TopN 板块，再拉取这些板块的成分股。
    它不会触发任何 LLM 分析，也不会调用完整交易图。

    Args:
        indicator: 资金流统计窗口，常用值为 今日、5日、10日。
        sector_type: 板块类型，支持 industry/行业 或 concept/概念。
        top_sectors: 选取资金流排名前多少个板块。
        stocks_per_sector: 每个板块最多纳入多少只股票。

    Returns:
        候选股票清单 CSV，包含来源板块、股票代码、股票名称等字段。
    """
    indicator = _normalize_indicator_input(indicator)
    fund_flow_text = route_to_vendor("get_sector_fund_flow", indicator, sector_type, top_sectors)
    fund_df = _tool_csv_to_df(fund_flow_text)
    if fund_df.empty:
        return "No sector fund-flow data available to build candidates."

    sector_col = _pick_sector_name_column(fund_df)
    if not sector_col:
        return f"Cannot identify sector name column in fund-flow data. Columns: {list(fund_df.columns)}"

    selected = fund_df.head(max(1, top_sectors)).copy()
    sector_names = [str(x).strip() for x in selected[sector_col].tolist() if str(x).strip()]

    candidate_rows = []
    for sector_name in sector_names:
        # 优先使用按主力资金净流入排序的板块个股资金流接口
        cons_text = route_to_vendor(
            "get_sector_stocks_fund_flow", sector_name, indicator, stocks_per_sector
        )
        cons_df = _tool_csv_to_df(cons_text)

        # 兜底：若新接口失败，回落到原成分股接口
        if cons_df.empty:
            cons_text = route_to_vendor(
                "get_sector_constituents", sector_name, sector_type, stocks_per_sector
            )
            cons_df = _tool_csv_to_df(cons_text)

        if cons_df.empty:
            continue

        ticker_col = _pick_ticker_column(cons_df)
        code_col = _pick_code_column(cons_df)
        name_col = _pick_sector_name_column(cons_df)

        if ticker_col is None:
            continue

        work_df = cons_df.head(max(1, stocks_per_sector)).copy()
        work_df["source_sector"] = sector_name
        work_df["ticker"] = work_df[ticker_col].astype(str).str.strip()
        work_df["stock_name"] = work_df[name_col].astype(str).str.strip() if name_col else ""
        work_df["raw_code"] = work_df[code_col].astype(str).str.strip() if code_col else ""

        # 透传主力净流入与涨跌幅（若可用），便于后续分析
        inflow_col = next(
            (c for c in cons_df.columns if "主力净流入" in c and ("净额" in c or c.endswith("净流入"))),
            None,
        )
        change_col = next((c for c in cons_df.columns if "涨跌幅" in c), None)
        price_col = next((c for c in cons_df.columns if "最新价" in c), None)
        work_df["main_net_inflow"] = work_df[inflow_col] if inflow_col else ""
        work_df["change_pct"] = work_df[change_col] if change_col else ""
        work_df["latest_price"] = work_df[price_col] if price_col else ""

        candidate_rows.append(
            work_df[
                [
                    "source_sector",
                    "ticker",
                    "stock_name",
                    "raw_code",
                    "main_net_inflow",
                    "change_pct",
                    "latest_price",
                ]
            ]
        )

    if not candidate_rows:
        lead_col = "领涨股" if "领涨股" in selected.columns else None
        if not lead_col:
            return "未能为所选板块构建候选股票清单。"

        out_df = pd.DataFrame(
            {
                "来源板块": sector_names,
                "股票代码": "",
                "股票名称": selected[lead_col].astype(str).str.strip().tolist(),
            }
        )
        sector_type_cn = "概念" if str(sector_type).strip().lower() in ("concept", "概念") else "行业"
        header = f"# 板块候选股票清单（资金流窗口={indicator}，板块类型={sector_type_cn}）\n"
        header += f"# 选中板块数: {len(sector_names)}\n"
        header += f"# 候选股票数: {len(out_df)}\n"
        header += "# 板块成分股接口不可用，已退化为各板块的领涨股。\n"
        header += "# 本结果仅为待分析候选清单，不会触发 LLM 分析。\n\n"
        return header + out_df.to_csv(index=False)

    out_df = pd.concat(candidate_rows, ignore_index=True)
    out_df = out_df.drop_duplicates(subset=["ticker", "source_sector"], keep="first")

    # 可选：补充前一交易日资金流
    if include_yesterday:
        prev_dates, prev_inflow, prev_change = [], [], []
        for ticker in out_df["ticker"].tolist():
            prev_text = route_to_vendor("get_stock_prev_day_fund_flow", ticker)
            prev_df = _tool_csv_to_df(prev_text)
            if prev_df.empty:
                prev_dates.append("")
                prev_inflow.append("")
                prev_change.append("")
                continue
            row = prev_df.iloc[0]
            prev_dates.append(str(row.get("date", "")).strip())
            prev_inflow.append(row.get("main_net_inflow", ""))
            prev_change.append(row.get("change_pct", ""))
        out_df["prev_date"] = prev_dates
        out_df["prev_main_net_inflow"] = prev_inflow
        out_df["prev_change_pct"] = prev_change

    # 根据开关选择保留的列
    base_cols = ["source_sector", "ticker", "stock_name"]
    today_cols = ["main_net_inflow", "change_pct", "latest_price"]
    yesterday_cols = ["prev_date", "prev_main_net_inflow", "prev_change_pct"]

    keep = list(base_cols)
    if include_today:
        keep += today_cols
    if include_yesterday:
        keep += yesterday_cols
    out_df = out_df[[c for c in keep if c in out_df.columns]]

    # 输出列名统一改为中文
    cn_rename = {
        "source_sector": "来源板块",
        "ticker": "股票代码",
        "stock_name": "股票名称",
        "main_net_inflow": "今日主力净流入",
        "change_pct": "今日涨跌幅(%)",
        "latest_price": "今日最新价",
        "prev_date": "前一交易日",
        "prev_main_net_inflow": "前日主力净流入",
        "prev_change_pct": "前日涨跌幅(%)",
    }
    out_df = out_df.rename(columns=cn_rename)

    sector_type_cn = "概念" if str(sector_type).strip().lower() in ("concept", "概念") else "行业"
    header = f"# 板块候选股票清单（资金流窗口={indicator}，板块类型={sector_type_cn}）\n"
    header += f"# 选中板块数: {len(sector_names)}\n"
    header += f"# 候选股票数: {len(out_df)}\n"
    header += f"# 包含今日={include_today}，包含前一交易日={include_yesterday}\n"
    header += "# 本结果仅为待分析候选清单，不会触发 LLM 分析。\n\n"
    return header + out_df.to_csv(index=False)
