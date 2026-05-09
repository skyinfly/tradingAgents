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
    try:
        ak = _load_akshare()
        candidates = ["news_economic_baidu", "news_global_cls", "news_cctv"]
        news_df = None
        for fn_name in candidates:
            fn = getattr(ak, fn_name, None)
            if fn is None:
                continue
            try:
                temp = fn()
                if temp is not None and not temp.empty:
                    news_df = temp
                    break
            except Exception:
                continue

        if news_df is None or news_df.empty:
            return "AKShare global news source unavailable."

        return f"## Global Market News, up to {curr_date}:\n\n{news_df.head(limit).to_csv(index=False)}"
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
