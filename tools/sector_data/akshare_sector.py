from datetime import datetime
from typing import Annotated

from tradingagents.dataflows.stockstats_utils import get_cached_text

_AKSHARE_PATCHED = False


def _patch_requests_default_ua():
    """给 requests 全局补默认 User-Agent。

    AKShare 部分接口（如 stock_sector_fund_flow_summary）调用 requests.get 时未带
    headers，在某些网络环境下（含本地 Clash 代理）服务端会直接关闭连接。
    这里在 Session.request 层注入默认 UA，覆盖所有底层调用。
    """
    global _AKSHARE_PATCHED
    if _AKSHARE_PATCHED:
        return
    import requests

    _orig_request = requests.Session.request

    def _patched_request(self, method, url, **kwargs):
        headers = kwargs.get("headers") or {}
        headers.setdefault(
            "User-Agent",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        )
        kwargs["headers"] = headers
        return _orig_request(self, method, url, **kwargs)

    requests.Session.request = _patched_request
    _AKSHARE_PATCHED = True


def _load_akshare():
    """延迟导入 AKShare，避免项目启动时强依赖数据源库。"""
    try:
        _patch_requests_default_ua()
        import akshare as ak
        return ak
    except Exception as e:
        raise RuntimeError(f"AKShare unavailable: {e}")


_EM_SECTOR_CODE_MAP_CACHE = None


def _get_em_sector_code_map(ak):
    """加载东方财富板块名→代码映射（带本地内存缓存与重试）。"""
    global _EM_SECTOR_CODE_MAP_CACHE
    if _EM_SECTOR_CODE_MAP_CACHE is not None:
        return _EM_SECTOR_CODE_MAP_CACHE
    import time
    from akshare.stock.stock_fund_em import _get_stock_sector_fund_flow_summary_code

    last_err = None
    for attempt in range(5):
        try:
            _EM_SECTOR_CODE_MAP_CACHE = _get_stock_sector_fund_flow_summary_code()
            return _EM_SECTOR_CODE_MAP_CACHE
        except Exception as e:
            last_err = e
            if attempt < 4:
                time.sleep(1.5 * (attempt + 1))
    raise last_err


def _infer_market(ticker: str) -> str:
    """根据 A 股代码推断交易所市场前缀（sh/sz/bj）。"""
    code = str(ticker).strip().zfill(6)
    if code.startswith(("60", "68", "9")):
        return "sh"
    if code.startswith(("00", "30", "20")):
        return "sz"
    if code.startswith(("8", "43", "92")):
        return "bj"
    return "sh"


def get_stock_prev_day_fund_flow_akshare(
    ticker: Annotated[str, "stock code, e.g. 000591"],
) -> str:
    """获取单股前一交易日的主力资金流（取最新两天里的上一行）。

    返回 CSV 文本，列：date, main_net_inflow, change_pct, close
    若数据不足或失败，返回 Error / No data 字符串。
    """
    cache_payload = {
        "ticker": str(ticker).strip().zfill(6),
        "date": datetime.now().strftime("%Y-%m-%d"),
    }
    return get_cached_text(
        "akshare_stock_prev_day_fund_flow",
        cache_payload,
        lambda: _get_stock_prev_day_fund_flow_akshare_uncached(ticker),
    )


def _get_stock_prev_day_fund_flow_akshare_uncached(ticker: str) -> str:
    """直接调用 AKShare 单股历史资金流接口，取倒数第二行作为前一日。"""
    import time

    try:
        ak = _load_akshare()
        code = str(ticker).strip().zfill(6)
        market = _infer_market(code)

        last_err = None
        df = None
        for attempt in range(5):
            try:
                df = ak.stock_individual_fund_flow(stock=code, market=market)
                break
            except Exception as e:
                last_err = e
                if attempt < 4:
                    time.sleep(min(1.5 * (attempt + 1), 6))
        if df is None:
            raise last_err

        if df is None or len(df) < 2:
            return f"No prev-day fund flow data for {ticker}"

        df_sorted = df.sort_values(df.columns[0]).reset_index(drop=True)
        prev_row = df_sorted.iloc[-2]
        out = {
            "date": str(prev_row.iloc[0]),
            "close": prev_row.get("收盘价", ""),
            "change_pct": prev_row.get("涨跌幅", ""),
            "main_net_inflow": prev_row.get("主力净流入-净额", ""),
        }
        header = f"# Prev-day fund flow for {code}\n"
        return header + ",".join(out.keys()) + "\n" + ",".join(str(v) for v in out.values())
    except Exception as e:
        return f"Error retrieving prev-day fund flow for {ticker}: {e}"


def _resolve_em_sector_name(ak, sector_name: str):
    """把同花顺/通用板块名映射到东方财富 fund_flow_summary 接口的板块名。

    顺序：精确匹配 → 前缀 → 包含 → 去掉常见后缀。
    """
    try:
        code_map = _get_em_sector_code_map(ak)
    except Exception:
        return sector_name  # 解析失败兜底为原名

    keys = list(code_map.keys())
    if sector_name in keys:
        return sector_name
    # 前缀匹配
    for k in keys:
        if k.startswith(sector_name) or sector_name.startswith(k):
            return k
    # 包含匹配（取最短的，避免过度泛化）
    contains = [k for k in keys if sector_name in k or k in sector_name]
    if contains:
        return min(contains, key=len)
    return None


def _normalize_sector_type(sector_type: str) -> str:
    """把用户输入的板块类型统一成内部使用的 industry/concept。"""
    st = (sector_type or "industry").strip().lower()
    if st in ("industry", "行业", "industry_fund"):
        return "industry"
    if st in ("concept", "概念", "concept_fund"):
        return "concept"
    return st


def _fund_sector_type_text(sector_type: str) -> str:
    """转换成 AKShare 资金流接口要求的中文板块类型。"""
    return "概念资金流" if _normalize_sector_type(sector_type) == "concept" else "行业资金流"


def _ths_fund_flow_window(indicator: str) -> str:
    """转换成同花顺资金流接口要求的统计窗口。"""
    value = _normalize_indicator(indicator)
    if value == "5日":
        return "5日排行"
    if value == "10日":
        return "10日排行"
    return "即时"


def _normalize_indicator(indicator: str) -> str:
    """把资金流窗口参数统一成 AKShare 支持的中文值。"""
    value = (indicator or "今日").strip()
    aliases = {
        "today": "今日",
        "day": "今日",
        "1d": "今日",
        "5d": "5日",
        "5day": "5日",
        "5days": "5日",
        "10d": "10日",
        "10day": "10日",
        "10days": "10日",
        "??": "今日",
        "5?": "5日",
        "10?": "10日",
    }
    return aliases.get(value.lower(), value)


def get_market_sectors_akshare(
    sector_type: Annotated[str, "industry or concept"] = "industry",
) -> str:
    """获取当前市场板块列表，并按板块类型缓存结果。"""
    cache_payload = {"sector_type": sector_type}
    return get_cached_text(
        "akshare_market_sectors",
        cache_payload,
        lambda: _get_market_sectors_akshare_uncached(sector_type),
    )


def _get_market_sectors_akshare_uncached(sector_type: str = "industry") -> str:
    """直接调用 AKShare 获取行业或概念板块列表。"""
    try:
        ak = _load_akshare()
        st = _normalize_sector_type(sector_type)
        if st == "concept":
            df = ak.stock_board_concept_name_em()
            source = "stock_board_concept_name_em"
        else:
            df = ak.stock_board_industry_name_em()
            source = "stock_board_industry_name_em"

        if df is None or df.empty:
            return f"No sector data found for sector_type={sector_type}"

        header = f"# Market sectors ({st})\n"
        header += f"# Total sectors: {len(df)}\n"
        header += f"# Source: {source}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        return header + df.to_csv(index=False)
    except Exception as e:
        return f"Error retrieving market sectors from AKShare: {e}"


def get_sector_fund_flow_akshare(
    indicator: Annotated[str, "今日/5日/10日"] = "今日",
    sector_type: Annotated[str, "industry or concept"] = "industry",
    top_n: Annotated[int, "number of sectors to return"] = 50,
) -> str:
    """获取板块资金流排行，并限制返回前 top_n 个板块。"""
    cache_payload = {
        "indicator": _normalize_indicator(indicator),
        "sector_type": sector_type,
        "top_n": top_n,
    }
    return get_cached_text(
        "akshare_sector_fund_flow_rank",
        cache_payload,
        lambda: _get_sector_fund_flow_akshare_uncached(indicator, sector_type, top_n),
    )


def _get_sector_fund_flow_akshare_uncached(
    indicator: str = "今日",
    sector_type: str = "industry",
    top_n: int = 50,
) -> str:
    """直接调用 AKShare 获取板块资金流排行。"""
    try:
        ak = _load_akshare()
        indicator = _normalize_indicator(indicator)
        fund_type = _fund_sector_type_text(sector_type)
        source = "stock_sector_fund_flow_rank"

        try:
            df = ak.stock_sector_fund_flow_rank(indicator=indicator, sector_type=fund_type)
        except Exception:
            if _normalize_sector_type(sector_type) == "concept":
                df = ak.stock_fund_flow_concept(symbol=_ths_fund_flow_window(indicator))
                source = "stock_fund_flow_concept"
            else:
                df = ak.stock_fund_flow_industry(symbol=_ths_fund_flow_window(indicator))
                source = "stock_fund_flow_industry"

        if df is None or df.empty:
            return f"No sector fund flow data found for indicator={indicator}, sector_type={sector_type}"

        if top_n and top_n > 0:
            df = df.head(top_n)

        header = f"# Sector fund flow rank ({fund_type}, indicator={indicator})\n"
        header += f"# Total rows: {len(df)}\n"
        header += f"# Source: {source}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        return header + df.to_csv(index=False)
    except Exception as e:
        return f"Error retrieving sector fund flow from AKShare: {e}"


def get_sector_constituents_akshare(
    sector_name: Annotated[str, "sector name, e.g. 小金属 / 融资融券"],
    sector_type: Annotated[str, "industry or concept"] = "industry",
    top_n: Annotated[int, "number of constituents to return"] = 200,
) -> str:
    """获取指定板块的成分股，并缓存同一板块的查询结果。"""
    cache_payload = {
        "sector_name": sector_name,
        "sector_type": sector_type,
        "top_n": top_n,
    }
    return get_cached_text(
        "akshare_sector_constituents",
        cache_payload,
        lambda: _get_sector_constituents_akshare_uncached(sector_name, sector_type, top_n),
    )


def get_sector_stocks_fund_flow_akshare(
    sector_name: Annotated[str, "sector name, e.g. 半导体 / 银行"],
    indicator: Annotated[str, "今日/5日/10日"] = "今日",
    top_n: Annotated[int, "number of stocks to return (ranked by main fund inflow)"] = 10,
) -> str:
    """获取指定板块内个股的资金流明细（按主力净流入排序）。"""
    cache_payload = {
        "sector_name": sector_name,
        "indicator": _normalize_indicator(indicator),
        "top_n": top_n,
    }
    return get_cached_text(
        "akshare_sector_stocks_fund_flow",
        cache_payload,
        lambda: _get_sector_stocks_fund_flow_akshare_uncached(sector_name, indicator, top_n),
    )


def _get_sector_stocks_fund_flow_akshare_uncached(
    sector_name: str,
    indicator: str = "今日",
    top_n: int = 10,
) -> str:
    """直接调用 AKShare 获取板块内个股资金流明细。"""
    import time

    try:
        ak = _load_akshare()
        indicator = _normalize_indicator(indicator)

        # 东方财富的板块命名与同花顺不同，先解析其内部 code map 并做 fuzzy match
        resolved_name = _resolve_em_sector_name(ak, sector_name)
        if not resolved_name:
            return (
                f"No matching East Money sector for '{sector_name}'. "
                f"Try adjusting the input sector name."
            )

        # push2.eastmoney.com 间歇性 reset 连接，做 8 次重试
        last_err = None
        df = None
        for attempt in range(8):
            try:
                df = ak.stock_sector_fund_flow_summary(symbol=resolved_name, indicator=indicator)
                break
            except Exception as e:
                last_err = e
                if attempt < 7:
                    time.sleep(min(2 * (attempt + 1), 8))
        if df is None:
            raise last_err

        if df.empty:
            return f"No sector stocks fund flow data found for sector_name={sector_name}, indicator={indicator}"

        if top_n and top_n > 0:
            df = df.head(top_n)

        header = f"# Sector stocks fund flow ({sector_name}, indicator={indicator})\n"
        header += f"# Total rows: {len(df)}\n"
        header += f"# Source: stock_sector_fund_flow_summary\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        return header + df.to_csv(index=False)
    except Exception as e:
        return f"Error retrieving sector stocks fund flow from AKShare: {e}"


def _get_sector_constituents_akshare_uncached(
    sector_name: str,
    sector_type: str = "industry",
    top_n: int = 200,
) -> str:
    """直接调用 AKShare 获取行业或概念板块成分股。"""
    try:
        ak = _load_akshare()
        st = _normalize_sector_type(sector_type)
        if st == "concept":
            df = ak.stock_board_concept_cons_em(symbol=sector_name)
            source = "stock_board_concept_cons_em"
        else:
            df = ak.stock_board_industry_cons_em(symbol=sector_name)
            source = "stock_board_industry_cons_em"

        if df is None or df.empty:
            return f"No sector constituents found for sector_name={sector_name}, sector_type={sector_type}"

        if top_n and top_n > 0:
            df = df.head(top_n)

        header = f"# Sector constituents ({sector_name}, type={st})\n"
        header += f"# Total rows: {len(df)}\n"
        header += f"# Source: {source}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        return header + df.to_csv(index=False)
    except Exception as e:
        return f"Error retrieving sector constituents from AKShare: {e}"
