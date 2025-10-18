import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from decimal import Decimal, getcontext
import warnings
warnings.filterwarnings("ignore")

# Technical indicators
import ta

# Import functions from generate_btc_ready
from scripts.pipeline.generate_btc_ready import (
    load_trades,
    aggregate_ohlcv,
    add_indicators,
    add_stats,
    calc_max_future_price,
    calc_min_future_price,
    STEP_BACK_PERC,
    LOSS_PERC
)


def load_btc_ready(btc_ready_dir, date):
    """Load BTC ready file for a given date"""
    ready_path = Path(btc_ready_dir) / f"{date}.csv"
    if not ready_path.exists():
        raise FileNotFoundError(f"BTC ready file not found: {ready_path}")
    
    df = pd.read_csv(ready_path)
    df['ts'] = pd.to_datetime(df['ts'], utc=True)
    return df


def merge_with_btc_data(pair_df, btc_df, windows):
    """
    Merge pair ready data with BTC ready data.
    
    For each row in pair_df, find the closest BTC row where ts(BTC) <= ts(pair)
    and copy the relevant BTC fields with 'btc_' prefix.
    
    Args:
        pair_df: DataFrame with pair ready data (must have 'ts' column)
        btc_df: DataFrame with BTC ready data (must have 'ts' column)
        windows: List of time windows (e.g., ['5min', '15min', '1h', '4h', '1d'])
    
    Returns:
        DataFrame with pair data enriched with BTC fields
    """
    # Ensure both dataframes are sorted by timestamp
    pair_df = pair_df.sort_values('ts').reset_index(drop=True) if len(pair_df) > 0 else pair_df
    btc_df = btc_df.sort_values('ts').reset_index(drop=True)
    
    # Handle empty pair_df early
    if len(pair_df) == 0:
        # Just add empty BTC columns and return
        base_fields = ['open', 'high', 'low', 'close', 'volume']
        btc_columns = []
        for window in windows:
            for field in base_fields:
                col_name = f"{window}_{field}"
                if col_name in btc_df.columns:
                    btc_columns.append(col_name)
        future_fields = [
            'max_future_price', 'max_future_price_perc', 'max_future_price_date',
            'min_future_price', 'min_future_price_perc', 'min_future_price_date'
        ]
        for field in future_fields:
            if field in btc_df.columns:
                btc_columns.append(field)
        for col in btc_columns:
            pair_df[f"btc_{col}"] = np.nan
        return pair_df
    
    # Ensure both have timezone-aware timestamps for comparison
    if pair_df['ts'].dt.tz is None:
        pair_df['ts'] = pd.to_datetime(pair_df['ts'], utc=True)
    if btc_df['ts'].dt.tz is None:
        btc_df['ts'] = pd.to_datetime(btc_df['ts'], utc=True)
    
    # Fields to copy from BTC for each window
    base_fields = ['open', 'high', 'low', 'close', 'volume']
    
    # Build list of all BTC columns to copy (window-prefixed fields)
    btc_columns = []
    for window in windows:
        for field in base_fields:
            col_name = f"{window}_{field}"
            if col_name in btc_df.columns:
                btc_columns.append(col_name)
    
    # Add non-window-prefixed future price fields (only calculated once in BTC)
    future_fields = [
        'max_future_price', 'max_future_price_perc', 'max_future_price_date',
        'min_future_price', 'min_future_price_perc', 'min_future_price_date'
    ]
    for field in future_fields:
        if field in btc_df.columns:
            btc_columns.append(field)
    
    # Initialize BTC columns in pair_df with NaN
    for col in btc_columns:
        pair_df[f"btc_{col}"] = np.nan
    
    # For each pair row, find the closest BTC row with ts <= pair_ts
    # Use pandas Series instead of numpy array to preserve timezone info
    btc_ts_series = btc_df['ts']
    
    for idx, row in pair_df.iterrows():
        pair_ts = row['ts']
        
        # Find all BTC timestamps <= pair_ts
        valid_btc_mask = btc_ts_series <= pair_ts
        
        if valid_btc_mask.any():
            # Get the index of the closest (most recent) BTC timestamp
            valid_indices = btc_ts_series[valid_btc_mask].index
            closest_btc_idx = valid_indices[-1]  # Last valid index = most recent
            
            # Copy BTC data
            for col in btc_columns:
                btc_value = btc_df.loc[closest_btc_idx, col]
                pair_df.at[idx, f"btc_{col}"] = btc_value
    
    return pair_df


def process_day(pair, date, raw_dir, btc_ready_dir, output_dir, windows, overwrite=False):
    """
    Process one day of data for a given pair.
    
    Generates ready file with pair data enriched with BTC reference data.
    
    Args:
        pair: Trading pair symbol (e.g., 'ETHUSDT')
        date: Date string in YYYY-MM-DD format
        raw_dir: Directory containing raw trade data
        btc_ready_dir: Directory containing BTCUSDT ready files
        output_dir: Directory to save ready files
        windows: List of time windows to aggregate
        overwrite: Whether to overwrite existing files
    """
    output_path = Path(output_dir) / pair / f"{date}.csv"
    
    if output_path.exists() and not overwrite:
        print(f"Skipping {output_path} (already exists)")
        return
    
    # 1. Load raw pair trades
    day_path = Path(raw_dir) / pair / f"{date}.csv"
    if not day_path.exists():
        print(f"Raw data not found: {day_path}")
        return
    
    print(f"Processing {pair} {date}...")
    trades_df = load_trades(pair, str(day_path))
    
    if len(trades_df) == 0:
        print(f"No trades found for {pair} on {date}")
        return
    
    # 2. Process all time windows for the pair
    merged_dfs = []
    for window in windows:
        df = aggregate_ohlcv(trades_df.copy(), window)
        
        if len(df) == 0:
            continue
        
        df = add_indicators(df)
        df = add_stats(df)
        
        # Reset index to convert ts from index to column
        df = df.reset_index()
        
        # Rename columns with window prefix (except ts)
        rename_dict = {col: f"{window}_{col}" for col in df.columns if col != 'ts'}
        df = df.rename(columns=rename_dict)
        
        merged_dfs.append(df)
    
    if not merged_dfs:
        print(f"No aggregated data for {pair} on {date}")
        return
    
    # 3. Merge all windows for the pair
    merged = merged_dfs[0]
    for df in merged_dfs[1:]:
        merged = merged.merge(df, on='ts', how='outer')
    
    merged = merged.sort_values('ts')
    
    # 4. Remove duplicate columns (if any)
    merged = merged.loc[:, ~merged.columns.duplicated()]
    
    # 5. Fill NaN values with appropriate strategies
    for col in merged.columns:
        if col == 'ts':
            continue
        if 'volume' in col or 'return' in col:
            merged[col] = merged[col].fillna(0)
        else:
            merged[col] = merged[col].ffill()
    
    # 6. Add future price labels (only once, using 5min_close or first close column)
    if "5min_close" in merged.columns:
        prices = merged["5min_close"].values
        timestamps = merged["ts"].values
    else:
        close_col = [c for c in merged.columns if "close" in c][0]
        prices = merged[close_col].values
        timestamps = merged["ts"].values
    
    # Calculate max future price
    max_price, max_perc, max_date = calc_max_future_price(prices, timestamps)
    merged["max_future_price"] = max_price
    merged["max_future_price_perc"] = max_perc
    merged["max_future_price_date"] = max_date
    
    # Calculate min future price
    min_price, min_perc, min_date = calc_min_future_price(prices, timestamps)
    merged["min_future_price"] = min_price
    merged["min_future_price_perc"] = min_perc
    merged["min_future_price_date"] = min_date
    
    # 7. Load BTC ready data and merge
    try:
        btc_df = load_btc_ready(btc_ready_dir, date)
        merged = merge_with_btc_data(merged, btc_df, windows)
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print(f"Continuing without BTC reference data for {pair} on {date}")
    
    # 8. Sort columns: ts first, then alphabetically
    cols = ['ts'] + sorted([c for c in merged.columns if c != 'ts'])
    merged = merged[cols]
    
    # 9. Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists():
        print(f"Overwriting {output_path}")
    
    merged.to_csv(output_path, index=False)
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate ready files for a trading pair with BTC reference data"
    )
    parser.add_argument(
        "--pair",
        type=str,
        required=True,
        help="Trading pair symbol (e.g., ETHUSDT)"
    )
    parser.add_argument(
        "--from",
        dest="date_from",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--to",
        dest="date_to",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--windows",
        type=str,
        default="5min,15min,1h,4h,1d",
        help="Comma-separated list of time windows (default: 5min,15min,1h,4h,1d)"
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Directory containing raw trade data (default: data/raw)"
    )
    parser.add_argument(
        "--btc-ready-dir",
        type=str,
        default="data/ready/BTCUSDT",
        help="Directory containing BTCUSDT ready files (default: data/ready/BTCUSDT)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/ready",
        help="Output directory for ready files (default: data/ready)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip processing if output file already exists"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite of existing files"
    )
    
    args = parser.parse_args()
    
    # Parse dates
    date_from = datetime.strptime(args.date_from, "%Y-%m-%d")
    date_to = datetime.strptime(args.date_to, "%Y-%m-%d")
    
    # Parse windows
    windows = [w.strip() for w in args.windows.split(",")]
    
    # Process each day
    current_date = date_from
    while current_date <= date_to:
        date_str = current_date.strftime("%Y-%m-%d")
        
        try:
            process_day(
                pair=args.pair,
                date=date_str,
                raw_dir=args.raw_dir,
                btc_ready_dir=args.btc_ready_dir,
                output_dir=args.output_dir,
                windows=windows,
                overwrite=args.force
            )
        except Exception as e:
            print(f"Error processing {args.pair} on {date_str}: {e}")
            import traceback
            traceback.print_exc()
        
        current_date += timedelta(days=1)


if __name__ == "__main__":
    main()
