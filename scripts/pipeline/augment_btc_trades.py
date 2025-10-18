import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# Technical indicators
import ta

def load_trades(pair, day_path):
    df = pd.read_csv(day_path, parse_dates=["ts"])
    df = df.sort_values("ts")
    df["price"] = df["price"].astype(float)
    df["qty"] = df["qty"].astype(float)
    return df

def aggregate_ohlcv(df, window):
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format='mixed')
    df = df.set_index("ts")
    ohlcv = df["price"].resample(window).ohlc()
    ohlcv["volume"] = df["qty"].resample(window).sum()
    ohlcv = ohlcv.dropna()
    return ohlcv

def add_indicators(df):
    n = len(df)
    # SMA, EMA, Bollinger Bands (window=20)
    if n >= 20:
        df["sma_20"] = ta.trend.sma_indicator(df["close"], window=20)
        df["ema_20"] = ta.trend.ema_indicator(df["close"], window=20)
        bb = ta.volatility.BollingerBands(df["close"], window=20)
        df["bb_high"] = bb.bollinger_hband()
        df["bb_low"] = bb.bollinger_lband()
    # RSI, ATR, Stochastic (window=14)
    if n >= 15:
        df["rsi_14"] = ta.momentum.rsi(df["close"], window=14)
        df["atr_14"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
        df["stoch_k"] = ta.momentum.stoch(df["high"], df["low"], df["close"], window=14)
        df["stoch_d"] = ta.momentum.stoch_signal(df["high"], df["low"], df["close"], window=14)
    # ADX (window=14) requires at least 2*window-1 data points
    if n >= 27:
        df["adx_14"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
    # MACD (window_slow=26)
    if n >= 35:
        df["macd"] = ta.trend.macd(df["close"])
        df["macd_signal"] = ta.trend.macd_signal(df["close"])
    return df

def add_stats(df):
    # Returns and volatility
    df["return"] = df["close"].pct_change()
    df["volatility"] = df["return"].rolling(20).std()
    # Drawdown
    df["cummax"] = df["close"].cummax()
    df["drawdown"] = (df["close"] - df["cummax"]) / df["cummax"]
    # Sharpe and Sortino (annualized)
    df["sharpe"] = df["return"].rolling(20).mean() / df["return"].rolling(20).std() * np.sqrt(252)
    df["sortino"] = df["return"].rolling(20).mean() / df["return"].rolling(20).std(ddof=0) * np.sqrt(252)
    # Skewness and kurtosis
    df["skew"] = df["return"].rolling(20).skew()
    df["kurtosis"] = df["return"].rolling(20).kurt()
    return df

def process_day(pair, day_path, out_dir, windows):
    df = load_trades(pair, day_path)
    dfs = []
    for window in windows:
        ohlcv = aggregate_ohlcv(df, window)
        ohlcv = add_indicators(ohlcv)
        ohlcv = add_stats(ohlcv)
        ohlcv = ohlcv.reset_index()  # Ensure 'ts' is a column before prefixing
        # Rename columns to include window prefix, except for 'ts'
        ohlcv = ohlcv.add_prefix(f"{window}_")
        ohlcv = ohlcv.rename(columns={f"{window}_ts": "ts"})
        dfs.append(ohlcv)
    # Merge all windows on 'ts' (outer join)
    merged = dfs[0]
    for d in dfs[1:]:
        merged = pd.merge(merged, d, on="ts", how="outer")
    merged = merged.sort_values("ts").reset_index(drop=True)
    
    # Remove duplicate columns (keep first occurrence)
    merged = merged.loc[:, ~merged.columns.duplicated()]
    
    # Ensure 'ts' is the first column
    if 'ts' in merged.columns:
        cols = ['ts'] + [c for c in merged.columns if c != 'ts']
        merged = merged[cols]

    # Fill NaN values with statistically meaningful defaults
    # Identify columns AFTER removing duplicates
    all_cols = list(merged.columns)
    price_cols = [c for c in all_cols if any(x in c for x in ["open","high","low","close"])]
    volume_cols = [c for c in all_cols if "volume" in c]
    indicator_cols = [c for c in all_cols if any(x in c for x in ["sma","ema","bb_","rsi","atr","stoch","adx","macd","signal"])]
    returns_cols = [c for c in all_cols if "return" in c or "volatility" in c or "drawdown" in c or "sharpe" in c or "sortino" in c or "skew" in c or "kurtosis" in c]

    # Fill NaN column by column to avoid shape mismatch errors
    for col in price_cols + indicator_cols:
        merged[col] = merged[col].ffill().bfill()
    for col in volume_cols:
        merged[col] = merged[col].fillna(0)
    for col in returns_cols:
        merged[col] = merged[col].fillna(0)
    out_path = out_dir / f"{day_path.stem}.csv"
    # Handle skip_existing and force flags
    skip_existing = getattr(process_day, "skip_existing", False)
    force = getattr(process_day, "force", False)
    if out_path.exists():
        if skip_existing and not force:
            print(f"Skipping {out_path} (already exists)")
            return
        if force:
            print(f"Overwriting {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"Saved merged {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Augment raw trades with technical/statistical indicators.")
    parser.add_argument("--pair", required=True)
    parser.add_argument("--from", dest="from_date", required=True)
    parser.add_argument("--to", dest="to_date", required=True)
    parser.add_argument("--windows", nargs="*", default=["5min","15min","1h","4h","1d"])
    parser.add_argument("--skip-existing", action="store_true", help="Skip days already processed if output exists")
    parser.add_argument("--force", action="store_true", help="Force regeneration and overwrite output files")
    args = parser.parse_args()

    pair = args.pair.upper()
    from_dt = datetime.strptime(args.from_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    to_dt = datetime.strptime(args.to_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    in_dir = Path(f"data/raw/{pair}")
    out_dir = Path(f"data/augmented/{pair}")

    current = from_dt
    while current < to_dt:
        day_path = in_dir / f"{current.strftime('%Y-%m-%d')}.csv"
        if day_path.exists():
            # Set skip_existing flag for process_day
            process_day.skip_existing = args.skip_existing
            process_day.force = args.force
            process_day(pair, day_path, out_dir, args.windows)
        else:
            print(f"No raw data for {day_path}")
        current += timedelta(days=1)

if __name__ == "__main__":
    main()
