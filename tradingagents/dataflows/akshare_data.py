from datetime import datetime, timedelta
from typing import Annotated

import pandas as pd

from .stockstats_utils import get_cached_text


def _normalize_symbol(symbol: str) -> str:
    s = symbol.strip().upper()
    if s.startswith(("SH", "SZ")) and len(s) >= 8:
        s = s[-6:]
    if "." in s:
        s = s.split(".", 1)[0]
    return s


def _to_ak_date(date_str: str) -> str:
    return datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y%m%d")


def _to_em_symbol(symbol: str) -> str:
    code = _normalize_symbol(symbol)
    if code.startswith("6") or code.startswith("9"):
        return f"SH{code}"
    if code.startswith("0") or code.startswith("2") or code.startswith("3"):
        return f"SZ{code}"
    return code


def _load_akshare():
    try:
        import akshare as ak
        return ak
    except Exception as e:
        raise RuntimeError(f"AKShare unavailable: {e}")


def _ak_hist_df(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    ak = _load_akshare()
    data = ak.stock_zh_a_hist(
        symbol=_normalize_symbol(symbol),
        period="daily",
        start_date=_to_ak_date(start_date),
        end_date=_to_ak_date(end_date),
        adjust="qfq",
    )
    if data is None or data.empty:
        return pd.DataFrame()

    keep_cols = ["日期", "开盘", "最高", "最低", "收盘", "成交量"]
    available = [c for c in keep_cols if c in data.columns]
    out = data[available].copy()
    rename_map = {
        "日期": "Date",
        "开盘": "Open",
        "最高": "High",
        "最低": "Low",
        "收盘": "Close",
        "成交量": "Volume",
    }
    out = out.rename(columns=rename_map)
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["Date", "Close"]).sort_values("Date")
    return out


def get_stock_data_akshare(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
):
    cache_payload = {
        "symbol": _normalize_symbol(symbol),
        "start_date": start_date,
        "end_date": end_date,
    }
    return get_cached_text(
        "akshare_stock_data",
        cache_payload,
        lambda: _get_stock_data_akshare_uncached(symbol, start_date, end_date),
    )


def _get_stock_data_akshare_uncached(symbol: str, start_date: str, end_date: str) -> str:
    try:
        data = _ak_hist_df(symbol, start_date, end_date)
        if data.empty:
            return f"No AKShare data found for symbol '{symbol}' between {start_date} and {end_date}"

        header = f"# Stock data for {_normalize_symbol(symbol)} from {start_date} to {end_date}\n"
        header += f"# Total records: {len(data)}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        return header + data.to_csv(index=False)
    except Exception as e:
        return f"Error retrieving AKShare stock data for {symbol}: {e}"


def get_indicators_akshare(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    from stockstats import wrap

    supported = {
        "close_50_sma", "close_200_sma", "close_10_ema", "macd", "macds",
        "macdh", "rsi", "boll", "boll_ub", "boll_lb", "atr", "vwma", "mfi",
    }
    if indicator not in supported:
        return f"Indicator {indicator} is not supported in AKShare vendor."

    try:
        end_dt = datetime.strptime(curr_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=max(look_back_days + 260, 400))
        data = _ak_hist_df(symbol, start_dt.strftime("%Y-%m-%d"), curr_date)
        if data.empty:
            return f"No AKShare data found for indicator {indicator} on {symbol}"

        df = wrap(data.copy())
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df[indicator]

        before = end_dt - timedelta(days=look_back_days)
        lines = []
        cur = end_dt
        while cur >= before:
            d = cur.strftime("%Y-%m-%d")
            row = df[df["Date"] == d]
            if row.empty:
                lines.append(f"{d}: N/A: Not a trading day (weekend or holiday)")
            else:
                val = row[indicator].iloc[0]
                lines.append(f"{d}: {'N/A' if pd.isna(val) else val}")
            cur -= timedelta(days=1)

        return f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {curr_date}:\n\n" + "\n".join(lines)
    except Exception as e:
        return f"Error retrieving AKShare indicator {indicator} for {symbol}: {e}"


def get_news_akshare(ticker: str, start_date: str, end_date: str) -> str:
    cache_payload = {
        "ticker": _normalize_symbol(ticker),
        "start_date": start_date,
        "end_date": end_date,
    }
    return get_cached_text(
        "akshare_news",
        cache_payload,
        lambda: _get_news_akshare_uncached(ticker, start_date, end_date),
    )


def _get_news_akshare_uncached(ticker: str, start_date: str, end_date: str) -> str:
    try:
        ak = _load_akshare()
        news_df = ak.stock_news_em(symbol=_normalize_symbol(ticker))
        if news_df is None or news_df.empty:
            return f"No news found for {ticker}"

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        if "发布时间" in news_df.columns:
            news_df["发布时间"] = pd.to_datetime(news_df["发布时间"], errors="coerce")
            news_df = news_df[(news_df["发布时间"] >= start_dt) & (news_df["发布时间"] <= end_dt + timedelta(days=1))]

        lines = []
        for _, row in news_df.head(30).iterrows():
            title = row.get("新闻标题", "No title")
            content = row.get("新闻内容", "")
            source = row.get("文章来源", "Unknown")
            link = row.get("新闻链接", "")
            lines.append(f"### {title} (source: {source})")
            if isinstance(content, str) and content.strip():
                lines.append(content.strip())
            if isinstance(link, str) and link.strip():
                lines.append(f"Link: {link.strip()}")
            lines.append("")

        if not lines:
            return f"No news found for {ticker} between {start_date} and {end_date}"
        return f"## {ticker} News, from {start_date} to {end_date}:\n\n" + "\n".join(lines)
    except Exception as e:
        return f"Error fetching AKShare news for {ticker}: {e}"


def get_global_news_akshare(curr_date: str, look_back_days: int = 7, limit: int = 10) -> str:
    cache_payload = {"curr_date": curr_date, "look_back_days": look_back_days, "limit": limit}
    return get_cached_text(
        "akshare_global_news",
        cache_payload,
        lambda: _get_global_news_akshare_uncached(curr_date, look_back_days, limit),
    )


def _get_global_news_akshare_uncached(curr_date: str, look_back_days: int = 7, limit: int = 10) -> str:
    """获取截至 curr_date、回看 look_back_days 天的宏观新闻。

    AKShare 的 ``news_cctv`` 与 ``news_economic_baidu`` 都接受 ``date=YYYYMMDD``
    参数，默认值是函数定义日期（很可能是几年前）。必须显式传日期并按时间倒序，
    否则拿到的是历史快照。
    """
    try:
        ak = _load_akshare()
        end_dt = datetime.strptime(curr_date, "%Y-%m-%d")

        # 优先：央视新闻联播（每日一组宏观要闻，按 date 字段倒序）
        fn_cctv = getattr(ak, "news_cctv", None)
        if fn_cctv is not None:
            collected = []
            for delta in range(look_back_days + 1):
                d = end_dt - timedelta(days=delta)
                try:
                    df = fn_cctv(date=d.strftime("%Y%m%d"))
                    if df is not None and not df.empty:
                        collected.append(df)
                except Exception:
                    continue
                if sum(len(x) for x in collected) >= limit:
                    break
            if collected:
                combined = pd.concat(collected, ignore_index=True)
                if "date" in combined.columns:
                    combined["date"] = combined["date"].astype(str)
                    combined = combined.sort_values("date", ascending=False)
                combined = combined.head(limit)
                header = (
                    f"## 全球宏观新闻 (截至 {curr_date}，回看 {look_back_days} 天，来源: 央视新闻联播)\n"
                    f"# 条数: {len(combined)}\n\n"
                )
                return header + combined.to_csv(index=False)

        # 次选：百度经济数据日历（含美/中/欧重要数据，按 日期+时间 倒序）
        fn_baidu = getattr(ak, "news_economic_baidu", None)
        if fn_baidu is not None:
            collected = []
            for delta in range(look_back_days + 1):
                d = end_dt - timedelta(days=delta)
                try:
                    df = fn_baidu(date=d.strftime("%Y%m%d"))
                    if df is not None and not df.empty:
                        collected.append(df)
                except Exception:
                    continue
                if sum(len(x) for x in collected) >= limit * 3:
                    break
            if collected:
                combined = pd.concat(collected, ignore_index=True)
                sort_cols = [c for c in ("日期", "时间") if c in combined.columns]
                if sort_cols:
                    combined = combined.sort_values(sort_cols, ascending=False)
                combined = combined.head(limit)
                header = (
                    f"## 全球经济日历 (截至 {curr_date}，回看 {look_back_days} 天，来源: 百度财经)\n"
                    f"# 条数: {len(combined)}\n\n"
                )
                return header + combined.to_csv(index=False)

        return "AKShare 全球新闻数据源均不可用。"
    except Exception as e:
        return f"Error fetching AKShare global news: {e}"


def get_fundamentals_akshare(ticker: str, curr_date: str = None) -> str:
    cache_payload = {"ticker": _normalize_symbol(ticker), "curr_date": curr_date}
    return get_cached_text(
        "akshare_fundamentals",
        cache_payload,
        lambda: _get_fundamentals_akshare_uncached(ticker, curr_date),
    )


def get_balance_sheet_akshare(ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
    cache_payload = {"ticker": _normalize_symbol(ticker), "freq": freq, "curr_date": curr_date}
    return get_cached_text(
        "akshare_balance_sheet",
        cache_payload,
        lambda: _get_balance_sheet_akshare_uncached(ticker, freq, curr_date),
    )


def get_cashflow_akshare(ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
    cache_payload = {"ticker": _normalize_symbol(ticker), "freq": freq, "curr_date": curr_date}
    return get_cached_text(
        "akshare_cashflow",
        cache_payload,
        lambda: _get_cashflow_akshare_uncached(ticker, freq, curr_date),
    )


def get_income_statement_akshare(ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
    cache_payload = {"ticker": _normalize_symbol(ticker), "freq": freq, "curr_date": curr_date}
    return get_cached_text(
        "akshare_income_statement",
        cache_payload,
        lambda: _get_income_statement_akshare_uncached(ticker, freq, curr_date),
    )


def get_insider_transactions_akshare(ticker: str) -> str:
    cache_payload = {"ticker": _normalize_symbol(ticker)}
    return get_cached_text(
        "akshare_insider_transactions",
        cache_payload,
        lambda: _get_insider_transactions_akshare_uncached(ticker),
    )


def _load_financial_sheet(ticker: str, fn_candidates, curr_date: str = None) -> pd.DataFrame:
    ak = _load_akshare()
    code = _normalize_symbol(ticker)
    em_symbol = _to_em_symbol(code)
    last_error = None
    symbol_try = [em_symbol, code]

    for fn_name in fn_candidates:
        fn = getattr(ak, fn_name, None)
        if fn is None:
            continue
        for sym in symbol_try:
            try:
                df = fn(symbol=sym)
                if df is None or df.empty:
                    continue
                if curr_date and "REPORT_DATE" in df.columns:
                    cutoff = pd.to_datetime(curr_date, errors="coerce")
                    report_dt = pd.to_datetime(df["REPORT_DATE"], errors="coerce")
                    df = df[report_dt <= cutoff]
                return df
            except Exception as e:
                last_error = e

    if last_error:
        raise RuntimeError(last_error)
    return pd.DataFrame()


def _get_fundamentals_akshare_uncached(ticker: str, curr_date: str = None) -> str:
    try:
        bs = _load_financial_sheet(
            ticker,
            ["stock_balance_sheet_by_report_em", "stock_balance_sheet_by_yearly_em"],
            curr_date=curr_date,
        )
        ps = _load_financial_sheet(
            ticker,
            ["stock_profit_sheet_by_report_em", "stock_profit_sheet_by_yearly_em", "stock_profit_sheet_by_quarterly_em"],
            curr_date=curr_date,
        )
        cf = _load_financial_sheet(
            ticker,
            ["stock_cash_flow_sheet_by_report_em", "stock_cash_flow_sheet_by_yearly_em", "stock_cash_flow_sheet_by_quarterly_em"],
            curr_date=curr_date,
        )

        if bs.empty and ps.empty and cf.empty:
            return f"No AKShare fundamentals found for {ticker}"

        chunks = [f"# Company Fundamentals for {_normalize_symbol(ticker)}"]
        chunks.append(f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if not bs.empty:
            chunks.append("## Balance Sheet (latest rows)")
            chunks.append(bs.head(5).to_csv(index=False))
        if not ps.empty:
            chunks.append("## Income Statement / Profit Sheet (latest rows)")
            chunks.append(ps.head(5).to_csv(index=False))
        if not cf.empty:
            chunks.append("## Cash Flow Sheet (latest rows)")
            chunks.append(cf.head(5).to_csv(index=False))
        return "\n".join(chunks)
    except Exception as e:
        return f"Error retrieving AKShare fundamentals for {ticker}: {e}"


def _get_balance_sheet_akshare_uncached(ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
    try:
        fn_candidates = (
            ["stock_balance_sheet_by_yearly_em"] if str(freq).lower() == "annual"
            else ["stock_balance_sheet_by_report_em", "stock_balance_sheet_by_yearly_em"]
        )
        df = _load_financial_sheet(ticker, fn_candidates, curr_date=curr_date)
        if df.empty:
            return f"No AKShare balance sheet data found for {ticker}"
        header = f"# Balance Sheet data for {_normalize_symbol(ticker)} ({freq})\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        return header + df.to_csv(index=False)
    except Exception as e:
        return f"Error retrieving AKShare balance sheet for {ticker}: {e}"


def _get_cashflow_akshare_uncached(ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
    try:
        fn_candidates = (
            ["stock_cash_flow_sheet_by_yearly_em"] if str(freq).lower() == "annual"
            else ["stock_cash_flow_sheet_by_report_em", "stock_cash_flow_sheet_by_quarterly_em", "stock_cash_flow_sheet_by_yearly_em"]
        )
        df = _load_financial_sheet(ticker, fn_candidates, curr_date=curr_date)
        if df.empty:
            return f"No AKShare cash flow data found for {ticker}"
        header = f"# Cash Flow data for {_normalize_symbol(ticker)} ({freq})\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        return header + df.to_csv(index=False)
    except Exception as e:
        return f"Error retrieving AKShare cash flow for {ticker}: {e}"


def _get_income_statement_akshare_uncached(ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
    try:
        fn_candidates = (
            ["stock_profit_sheet_by_yearly_em"] if str(freq).lower() == "annual"
            else ["stock_profit_sheet_by_report_em", "stock_profit_sheet_by_quarterly_em", "stock_profit_sheet_by_yearly_em"]
        )
        df = _load_financial_sheet(ticker, fn_candidates, curr_date=curr_date)
        if df.empty:
            return f"No AKShare income statement data found for {ticker}"
        header = f"# Income Statement data for {_normalize_symbol(ticker)} ({freq})\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        return header + df.to_csv(index=False)
    except Exception as e:
        return f"Error retrieving AKShare income statement for {ticker}: {e}"


def _get_insider_transactions_akshare_uncached(ticker: str) -> str:
    try:
        ak = _load_akshare()
        code = _normalize_symbol(ticker)
        em_symbol = _to_em_symbol(code)

        # Some AKShare versions expose stock_ggcg_em with different symbol expectations.
        fn = getattr(ak, "stock_ggcg_em", None)
        if fn is None:
            return "AKShare insider transactions interface (stock_ggcg_em) is unavailable."

        df = pd.DataFrame()
        errors = []
        for sym in [em_symbol, code]:
            try:
                temp = fn(symbol=sym)
                if temp is not None and not temp.empty:
                    df = temp
                    break
            except Exception as e:
                errors.append(str(e))

        if df.empty:
            err = f" ({errors[-1]})" if errors else ""
            return f"No AKShare insider transactions data found for {ticker}{err}"

        header = f"# Insider Transactions data for {code}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        return header + df.to_csv(index=False)
    except Exception as e:
        return f"Error retrieving AKShare insider transactions for {ticker}: {e}"


# ==================== A 股盘中实时数据接口 ====================
# 不缓存：盘中数据本质上随时变化，缓存反而误事。仅在网络层加重试。

def _retry_call(fn, attempts=5, base_delay=1.5):
    """对 AKShare 接口做带退避的重试，吸收 push2.eastmoney.com 间歇抖动。"""
    import time as _t
    last_err = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if i < attempts - 1:
                _t.sleep(min(base_delay * (i + 1), 6))
    if last_err:
        raise last_err
    return None


def _akshare_with_ua():
    """确保 requests 默认 UA 已注入（盘中接口对 UA 敏感），并返回 ak。"""
    from tools.sector_data.akshare_sector import _patch_requests_default_ua
    _patch_requests_default_ua()
    import akshare as ak
    return ak


def _market_prefix(code: str) -> str:
    c = str(code).strip().zfill(6)
    if c.startswith(("60", "68", "9")):
        return "sh"
    if c.startswith(("00", "30", "20")):
        return "sz"
    return "sh"


def get_realtime_quote_akshare(
    symbol: Annotated[str, "A 股代码，如 002031"],
) -> str:
    """获取个股实时报价 + 五档盘口（盘中刷新）。"""
    try:
        ak = _akshare_with_ua()
        code = _normalize_symbol(symbol)

        # stock_bid_ask_em 返回 long format：item / value
        df = _retry_call(lambda: ak.stock_bid_ask_em(symbol=code))
        if df is None or df.empty:
            return f"No realtime quote available for {symbol}"

        # 转 long -> 平铺 dict
        snap = dict(zip(df["item"].astype(str), df["value"]))

        # 五档（卖 5 在上，买 1 在下，便于阅读）
        bidask_lines = ["五档盘口 (price / volume)"]
        for side, label in [("sell", "卖"), ("buy", "买")]:
            levels = range(5, 0, -1) if side == "sell" else range(1, 6)
            for lv in levels:
                p = snap.get(f"{side}_{lv}", "-")
                v = snap.get(f"{side}_{lv}_vol", "-")
                bidask_lines.append(f"  {label}{lv}:  {p}  /  {v}")

        # 关键报价字段（AKShare 这里字段名都是英文 snake_case）
        key_fields = [
            ("最新价", "latest_price"),
            ("涨跌额", "price_change"),
            ("涨跌幅(%)", "change_percent"),
            ("今开", "open_price"),
            ("最高", "high_price"),
            ("最低", "low_price"),
            ("昨收", "previous_close"),
            ("成交量(手)", "volume"),
            ("成交额(元)", "amount"),
            ("换手率(%)", "turnover_rate"),
            ("量比", "volume_ratio"),
            ("委比(%)", "commission_ratio"),
            ("委差", "commission_difference"),
        ]
        kv_lines = ["关键报价"]
        for cn_label, key in key_fields:
            if key in snap:
                kv_lines.append(f"  {cn_label}: {snap[key]}")

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = (
            f"# 实时报价 - {code}\n"
            f"# 数据来源: stock_bid_ask_em (东方财富)\n"
            f"# 拉取时间: {ts}\n"
            f"# 注意: 盘中数据约 3-5 秒延迟；非交易时段五档显示为 '-'。\n\n"
        )
        return header + "\n".join(kv_lines) + "\n\n" + "\n".join(bidask_lines)
    except Exception as e:
        return f"Error retrieving realtime quote for {symbol}: {e}"


def get_intraday_minute_bars_akshare(
    symbol: Annotated[str, "A 股代码，如 002031"],
    period: Annotated[str, "分钟级别: 1/5/15/30/60"] = "5",
    lookback_minutes: Annotated[int, "回看分钟数（含今日盘中）"] = 240,
) -> str:
    """获取个股最近 N 分钟的分钟 K 线（含今日盘中）。"""
    try:
        ak = _akshare_with_ua()
        code = _normalize_symbol(symbol)

        end_dt = datetime.now()
        start_dt = end_dt - timedelta(minutes=int(lookback_minutes) + 30)

        start_s = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        end_s = end_dt.strftime("%Y-%m-%d %H:%M:%S")

        df = _retry_call(lambda: ak.stock_zh_a_hist_min_em(
            symbol=code, start_date=start_s, end_date=end_s,
            period=str(period), adjust="qfq",
        ))
        if df is None or df.empty:
            return f"No intraday minute bars available for {symbol} (period={period})"

        # 按时间列升序（如有）
        time_col = next((c for c in df.columns if "时间" in c or c.lower() == "datetime"), None)
        if time_col:
            df = df.sort_values(time_col)

        header = (
            f"# {code} 分钟 K 线 ({period} 分钟)\n"
            f"# 区间: {start_s} 到 {end_s}\n"
            f"# 条数: {len(df)}\n"
            f"# 数据来源: stock_zh_a_hist_min_em (东方财富)\n\n"
        )
        return header + df.to_csv(index=False)
    except Exception as e:
        return f"Error retrieving intraday minute bars for {symbol}: {e}"


def get_today_fund_flow_rank_akshare(
    top_n: Annotated[int, "返回前 N 名（按主力净流入降序）"] = 30,
    direction: Annotated[str, "inflow(净流入) 或 outflow(净流出)"] = "inflow",
) -> str:
    """获取全市场个股今日实时主力资金流排名（盘中累计）。"""
    try:
        ak = _akshare_with_ua()
        df = _retry_call(lambda: ak.stock_individual_fund_flow_rank(indicator="今日"))
        if df is None or df.empty:
            return "No today fund flow rank available."

        # 找到主力净流入列
        inflow_col = next(
            (c for c in df.columns if "主力" in c and ("净流入" in c) and ("净额" in c or c.endswith("净流入"))),
            None,
        )
        # 退化：寻找仅含"主力净流入"
        if inflow_col is None:
            inflow_col = next((c for c in df.columns if "主力" in c and "净流入" in c), None)

        if inflow_col is not None:
            df[inflow_col] = pd.to_numeric(df[inflow_col], errors="coerce")
            ascending = direction.lower() == "outflow"
            df = df.sort_values(inflow_col, ascending=ascending, na_position="last")
        df = df.head(max(1, int(top_n)))

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        direction_label = "净流出" if direction.lower() == "outflow" else "净流入"
        header = (
            f"# 全市场今日实时资金流排名 (Top {top_n}, 按主力{direction_label}降序)\n"
            f"# 拉取时间: {ts}\n"
            f"# 数据来源: stock_individual_fund_flow_rank(indicator='今日')\n"
            f"# 注意: 盘中数据约 30s 延迟；非交易时段为最近一个交易日累计。\n\n"
        )
        return header + df.to_csv(index=False)
    except Exception as e:
        return f"Error retrieving today fund flow rank: {e}"
