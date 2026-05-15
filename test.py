import time
from datetime import datetime
from io import StringIO
from pathlib import Path

import pandas as pd

from tools.sector_data.sector_tools import build_sector_stock_candidates

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 260)
pd.set_option("display.unicode.ambiguous_as_wide", True)
pd.set_option("display.unicode.east_asian_width", True)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "sector_candidates"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_case(title: str, params: dict, file_tag: str):
    print(f"\n{'#' * 70}\n# {title}\n# 参数: {params}\n{'#' * 70}")
    start = time.time()
    result = build_sector_stock_candidates.invoke(params)
    elapsed = time.time() - start

    meta = [l for l in result.splitlines() if l.startswith("#")]
    csv_lines = [l for l in result.splitlines() if l and not l.startswith("#")]

    print(f"耗时: {elapsed:.2f}s")
    for l in meta:
        print(l)

    if not csv_lines:
        print("(无数据)")
        return

    df = pd.read_csv(
        StringIO("\n".join(csv_lines)),
        dtype={"股票代码": str, "前一交易日": str},
    )

    # 落盘 CSV：UTF-8 BOM 方便 Excel 直接打开
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"{stamp}_{file_tag}.csv"
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        for l in meta:
            f.write(l + "\n")
        f.write("\n")
        df.to_csv(f, index=False)

    print("\n" + "=" * 70)
    print(df.to_string(index=False))
    print(f"\n已保存: {out_path.relative_to(PROJECT_ROOT)}")


# 默认：只看今日
run_case(
    "Case 1: 仅今日资金流（默认）",
    {"indicator": "今日", "sector_type": "industry", "top_sectors": 3, "stocks_per_sector": 5},
    "today",
)

# 今日 + 前一日对比
run_case(
    "Case 2: 今日 + 前一交易日 对比",
    {
        "indicator": "今日",
        "sector_type": "industry",
        "top_sectors": 2,
        "stocks_per_sector": 5,
        "include_today": True,
        "include_yesterday": True,
    },
    "today_vs_prev",
)

# 只看前一日
run_case(
    "Case 3: 仅前一交易日",
    {
        "indicator": "今日",
        "sector_type": "industry",
        "top_sectors": 2,
        "stocks_per_sector": 5,
        "include_today": False,
        "include_yesterday": True,
    },
    "prev_only",
)
