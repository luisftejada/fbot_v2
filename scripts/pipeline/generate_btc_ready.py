import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import numpy as np
from decimal import Decimal, getcontext
import warnings
warnings.filterwarnings("ignore")

# Technical indicators
import ta

STEP_BACK_PERC = 0.01  # 1%
LOSS_PERC = 0.005      # 0.5%


def load_trades(pair, day_path):
    """Load raw trades from CSV file"""
    df = pd.read_csv(day_path, parse_dates=["ts"])
    df = df.sort_values("ts")
    df["price"] = df["price"].astype(float)
    df["qty"] = df["qty"].astype(float)
    return df


def aggregate_ohlcv(df, window):
    """Aggregate trades into OHLCV candles for a given time window"""
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format='mixed')
    df = df.set_index("ts")
    ohlcv = df["price"].resample(window).ohlc()
    ohlcv["volume"] = df["qty"].resample(window).sum()
    ohlcv = ohlcv.dropna()
    return ohlcv


def add_indicators(df):
    """Add technical indicators to OHLCV dataframe"""
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
    """Add statistical features to OHLCV dataframe"""
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


def calc_max_future_price(prices, timestamps):
    """Calculate maximum future price for each timestamp"""
    n = len(prices)
    max_price = [None] * n
    max_perc = [None] * n
    max_date = [None] * n
    getcontext().prec = 10
    for i in range(n):
        initial = Decimal(str(prices[i]))
        future_price = initial
        future_date = timestamps[i]
        done = False
        for j in range(i+1, n):
            price = Decimal(str(prices[j]))
            # If the price increases, update future_price
            if price > future_price:
                future_price = price
                future_date = timestamps[j]
            # If the price drops more than STEP_BACK% from future_price, stop
            if price < future_price * Decimal(str(1 - STEP_BACK_PERC)):
                max_price[i] = float(future_price)
                max_perc[i] = max(0.0, float((future_price / initial) - Decimal('1')))
                max_date[i] = future_date
                done = True
                break
            # If the price drops more than LOSS_PERC% from initial, stop with loss
            if price < initial * Decimal(str(1 - LOSS_PERC)):
                max_price[i] = float(initial)
                max_perc[i] = 0.0
                max_date[i] = timestamps[i]
                done = True
                break
        if not done:
            max_price[i] = float(future_price)
            max_perc[i] = max(0.0, float((future_price / initial) - Decimal('1')))
            max_date[i] = future_date
    return max_price, max_perc, max_date


def calc_min_future_price(prices, timestamps):
    """Calculate minimum future price for each timestamp"""
    n = len(prices)
    min_price = [None] * n
    min_perc = [None] * n
    min_date = [None] * n
    getcontext().prec = 10
    for i in range(n):
        initial = Decimal(str(prices[i]))
        future_price = initial
        future_date = timestamps[i]
        done = False
        for j in range(i+1, n):
            price = Decimal(str(prices[j]))
            # If the price decreases, update future_price
            if price < future_price:
                future_price = price
                future_date = timestamps[j]
            # If the price increases more than STEP_BACK% from future_price, stop
            if price > future_price * Decimal(str(1 + STEP_BACK_PERC)):
                min_price[i] = float(future_price)
                min_perc[i] = max(0.0, float((initial / future_price) - Decimal('1')))
                min_date[i] = future_date
                done = True
                break
            # If the price increases more than LOSS_PERC% from initial, stop with loss
            if price > initial * Decimal(str(1 + LOSS_PERC)):
                min_price[i] = 0.0
                min_perc[i] = 0.0
                min_date[i] = timestamps[i]
                done = True
                break
        if not done:
            min_price[i] = float(future_price)
            min_perc[i] = max(0.0, float((initial / future_price) - Decimal('1')))
            min_date[i] = future_date
    return min_price, min_perc, min_date


def process_day(pair, day_path, out_dir, windows):
    """Process a single day of trades: aggregate, add indicators, add future prices"""
    # Load and aggregate trades
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
    all_cols = list(merged.columns)
    price_cols = [c for c in all_cols if any(x in c for x in ["open","high","low","close"])]
    volume_cols = [c for c in all_cols if "volume" in c]
    indicator_cols = [c for c in all_cols if any(x in c for x in ["sma","ema","bb_","rsi","atr","stoch","adx","macd","signal"])]
    returns_cols = [c for c in all_cols if "return" in c or "volatility" in c or "drawdown" in c or "sharpe" in c or "sortino" in c or "skew" in c or "kurtosis" in c]

    # Fill NaN column by column
    for col in price_cols + indicator_cols:
        merged[col] = merged[col].ffill().bfill()
    for col in volume_cols:
        merged[col] = merged[col].fillna(0)
    for col in returns_cols:
        merged[col] = merged[col].fillna(0)
    
    # Add future price labels
    # Use 5min_close if available, otherwise use the first close column found
    if "5min_close" in merged.columns:
        prices = merged["5min_close"].values
        timestamps = merged["ts"].values
    else:
        close_col = [c for c in merged.columns if "close" in c][0]
        prices = merged[close_col].values
        timestamps = merged["ts"].values
    
    max_vals, max_perc, max_dates = calc_max_future_price(prices, timestamps)
    min_vals, min_perc, min_dates = calc_min_future_price(prices, timestamps)
    merged["max_future_price"] = max_vals
    merged["max_future_price_perc"] = max_perc
    merged["max_future_price_date"] = max_dates
    merged["min_future_price"] = min_vals
    merged["min_future_price_perc"] = min_perc
    merged["min_future_price_date"] = min_dates
    
    # Save to output
    out_path = out_dir / f"{day_path.stem}.csv"
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
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate ready BTC features with indicators and future price labels.")
    parser.add_argument("--pair", required=True, help="Trading pair (e.g. BTCUSDT)")
    parser.add_argument("--from", dest="from_date", required=True, help="Start date (YYYY-MM-DD), inclusive")
    parser.add_argument("--to", dest="to_date", required=True, help="End date (YYYY-MM-DD), exclusive")
    parser.add_argument("--windows", nargs="*", default=["5min","15min","1h","4h","1d"], help="Time windows for OHLCV aggregation")
    parser.add_argument("--skip-existing", action="store_true", help="Skip days already processed if output exists")
    parser.add_argument("--force", action="store_true", help="Force regeneration and overwrite output files")
    args = parser.parse_args()

    pair = args.pair.upper()
    from_dt = datetime.strptime(args.from_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    to_dt = datetime.strptime(args.to_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    in_dir = Path(f"data/raw/{pair}")
    out_dir = Path(f"data/ready/{pair}")

    current = from_dt
    while current < to_dt:
        day_path = in_dir / f"{current.strftime('%Y-%m-%d')}.csv"
        if day_path.exists():
            process_day.skip_existing = args.skip_existing
            process_day.force = args.force
            process_day(pair, day_path, out_dir, args.windows)
        else:
            print(f"No raw data for {day_path}")
        current += timedelta(days=1)


if __name__ == "__main__":
    main()
