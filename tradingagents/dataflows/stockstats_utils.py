import time
import logging
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError
from stockstats import wrap
from typing import Annotated
import os
from .config import get_config

logger = logging.getLogger(__name__)
_YF_REQUEST_SEQ = 0
_YF_LAST_REQUEST_TS = 0.0
_YF_COOLDOWN_UNTIL_TS = 0.0


def _cache_dir() -> Path:
    config = get_config()
    path = Path(config["data_cache_dir"]) / "yfinance_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_path(namespace: str, payload: dict, suffix: str) -> Path:
    key = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return _cache_dir() / f"{namespace}-{digest}{suffix}"


def get_cached_text(namespace: str, payload: dict, fetcher):
    """Return cached text for a yfinance fetch, or store the fresh result."""
    path = _cache_path(namespace, payload, ".txt")
    if path.exists():
        cached = path.read_text(encoding="utf-8")
        if cached.startswith("Error "):
            try:
                path.unlink()
            except OSError:
                pass
        else:
            return cached

    result = fetcher()
    if isinstance(result, str) and not result.startswith("Error "):
        path.write_text(result, encoding="utf-8")
    return result


def yf_retry(func, max_retries=3, base_delay=2.0, request_name="unknown", request_meta=None):
    """Execute a yfinance call with exponential backoff on rate limits.

    yfinance raises YFRateLimitError on HTTP 429 responses but does not
    retry them internally. This wrapper adds retry logic specifically
    for rate limits. Other exceptions propagate immediately.
    """
    meta_str = ""
    if request_meta:
        safe_meta = {k: v for k, v in request_meta.items() if v is not None}
        if safe_meta:
            meta_str = " " + " ".join([f"{k}={v}" for k, v in safe_meta.items()])

    config = get_config()
    min_interval = float(config.get("yfinance_min_request_interval_sec", 1.2))
    cooldown_on_limit = float(config.get("yfinance_cooldown_on_limit_sec", 12.0))

    for attempt in range(max_retries + 1):
        try:
            global _YF_REQUEST_SEQ, _YF_LAST_REQUEST_TS, _YF_COOLDOWN_UNTIL_TS

            now = time.time()
            if now < _YF_COOLDOWN_UNTIL_TS:
                cooldown_wait = _YF_COOLDOWN_UNTIL_TS - now
                print(f"[Yahoo冷却] wait={cooldown_wait:.1f}s api={request_name}{meta_str}")
                time.sleep(cooldown_wait)

            now = time.time()
            since_last = now - _YF_LAST_REQUEST_TS
            if _YF_LAST_REQUEST_TS > 0 and since_last < min_interval:
                throttle_wait = min_interval - since_last
                print(f"[Yahoo节流] wait={throttle_wait:.1f}s api={request_name}{meta_str}")
                time.sleep(throttle_wait)

            _YF_REQUEST_SEQ += 1
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[Yahoo请求] t={ts} seq={_YF_REQUEST_SEQ} api={request_name} try={attempt}/{max_retries}{meta_str}")
            result = func()
            _YF_LAST_REQUEST_TS = time.time()
            return result
        except YFRateLimitError:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                _YF_COOLDOWN_UNTIL_TS = max(_YF_COOLDOWN_UNTIL_TS, time.time() + cooldown_on_limit)
                logger.warning(
                    f"[Yahoo限流] api={request_name} next_retry={attempt + 1}/{max_retries} wait={delay:.0f}s{meta_str}"
                )
                time.sleep(delay)
            else:
                _YF_COOLDOWN_UNTIL_TS = max(_YF_COOLDOWN_UNTIL_TS, time.time() + cooldown_on_limit)
                logger.warning(
                    f"[Yahoo限流-放弃] api={request_name} retries_exhausted={max_retries}{meta_str}"
                )
                raise


def _clean_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize a stock DataFrame for stockstats: parse dates, drop invalid rows, fill price gaps."""
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"])

    price_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in data.columns]
    data[price_cols] = data[price_cols].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["Close"])
    data[price_cols] = data[price_cols].ffill().bfill()

    return data


def load_ohlcv(symbol: str, curr_date: str) -> pd.DataFrame:
    """Fetch OHLCV data with caching, filtered to prevent look-ahead bias.

    Downloads 15 years of data up to today and caches per symbol. On
    subsequent calls the cache is reused. Rows after curr_date are
    filtered out so backtests never see future prices.
    """
    config = get_config()
    curr_date_dt = pd.to_datetime(curr_date)

    start_date = curr_date_dt - pd.DateOffset(years=5)
    end_date = curr_date_dt + pd.DateOffset(days=1)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    data_file = _cache_path(
        "ohlcv",
        {"symbol": symbol.upper(), "curr_date": curr_date, "start": start_str, "end": end_str},
        ".csv",
    )

    if os.path.exists(data_file):
        data = pd.read_csv(data_file, on_bad_lines="skip")
    else:
        data = yf_retry(
            lambda: yf.download(
                symbol,
                start=start_str,
                end=end_str,
                multi_level_index=False,
                progress=False,
                auto_adjust=True,
            ),
            request_name="download_ohlcv",
            request_meta={"symbol": symbol.upper(), "start": start_str, "end": end_str},
        )
        data = data.reset_index()
        data.to_csv(data_file, index=False)

    data = _clean_dataframe(data)
    data = data[data["Date"] <= curr_date_dt]
    return data


def filter_financials_by_date(data: pd.DataFrame, curr_date: str) -> pd.DataFrame:
    """Drop financial statement columns (fiscal period timestamps) after curr_date."""
    if not curr_date or data.empty:
        return data
    cutoff = pd.Timestamp(curr_date)
    mask = pd.to_datetime(data.columns, errors="coerce") <= cutoff
    return data.loc[:, mask]


class StockstatsUtils:
    @staticmethod
    def get_stock_stats(
        symbol: Annotated[str, "ticker symbol for the company"],
        indicator: Annotated[str, "quantitative indicators based off of the stock data for the company"],
        curr_date: Annotated[str, "curr date for retrieving stock price data, YYYY-mm-dd"],
    ):
        data = load_ohlcv(symbol, curr_date)
        df = wrap(data)
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
        curr_date_str = pd.to_datetime(curr_date).strftime("%Y-%m-%d")

        df[indicator]
        matching_rows = df[df["Date"].str.startswith(curr_date_str)]

        if not matching_rows.empty:
            indicator_value = matching_rows[indicator].values[0]
            return indicator_value
        return "N/A: Not a trading day (weekend or holiday)"
