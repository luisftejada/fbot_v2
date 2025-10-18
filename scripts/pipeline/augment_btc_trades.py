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
        # Rename columns to include window prefix, except for 'ts'
        ohlcv = ohlcv.add_prefix(f"{window}_")
        ohlcv = ohlcv.rename(columns={f"{window}_ts": "ts"})
        dfs.append(ohlcv)
    # Merge all windows on 'ts' (outer join)
    merged = dfs[0]
    for d in dfs[1:]:
        merged = pd.merge(merged, d, on="ts", how="outer")
    merged = merged.sort_values("ts").reset_index(drop=True)
    out_path = out_dir / f"{day_path.stem}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"Saved merged {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Augment raw trades with technical/statistical indicators.")
    parser.add_argument("--pair", required=True)
    parser.add_argument("--from", dest="from_date", required=True)
    parser.add_argument("--to", dest="to_date", required=True)
    parser.add_argument("--windows", nargs="*", default=["5min","15min","1h","4h","1d"])
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
            process_day(pair, day_path, out_dir, args.windows)
        else:
            print(f"No raw data for {day_path}")
        current += timedelta(days=1)

if __name__ == "__main__":
    main()
