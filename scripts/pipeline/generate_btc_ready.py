import argparse
from pathlib import Path
import pandas as pd
from decimal import Decimal, getcontext

STEP_BACK_PERC = 0.01  # 1%
LOSS_PERC = 0.005      # 0.5%



def calc_max_future_price(prices, timestamps):
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

def process_file(in_path, out_path):
    df = pd.read_csv(in_path)
    # Use 5min_close if available, otherwise use the first close column found
    if "5min_close" in df.columns:
        prices = df["5min_close"].values
        timestamps = df["ts"].values
    else:
        close_col = [c for c in df.columns if "close" in c][0]
        prices = df[close_col].values
        timestamps = df["ts"].values
    max_vals, max_perc, max_dates = calc_max_future_price(prices, timestamps)
    min_vals, min_perc, min_dates = calc_min_future_price(prices, timestamps)
    df["max_future_price"] = max_vals
    df["max_future_price_perc"] = max_perc
    df["max_future_price_date"] = max_dates
    df["min_future_price"] = min_vals
    df["min_future_price_perc"] = min_perc
    df["min_future_price_date"] = min_dates
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate ready features with future price labels.")
    parser.add_argument("--pair", required=True, help="Trading pair, e.g. BTCUSDT")
    args = parser.parse_args()
    pair = args.pair.upper()
    in_dir = Path(f"data/augmented/{pair}")
    out_dir = Path(f"data/ready/{pair}")
    for in_file in sorted(in_dir.glob("*.csv")):
        out_file = out_dir / in_file.name
        process_file(in_file, out_file)

if __name__ == "__main__":
    main()
