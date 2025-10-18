import argparse
import os
from datetime import datetime, timedelta, timezone
import csv
from pathlib import Path

BASE_URL = "https://data.binance.vision/data/spot/daily/trades"
USER_AGENT = "binance-vision-downloader/1.0"

def download_daily_trades(symbol: str, date: datetime, timeout: int = 120, max_retries: int = 3):
    import requests
    import io
    import zipfile
    date_str = date.strftime("%Y-%m-%d")
    filename = f"{symbol}-trades-{date_str}.zip"
    url = f"{BASE_URL}/{symbol}/{filename}"
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout, stream=True)
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            zip_content = io.BytesIO(resp.content)
            trades = []
            with zipfile.ZipFile(zip_content) as zf:
                csv_filename = filename.replace(".zip", ".csv")
                with zf.open(csv_filename) as csv_file:
                    csv_text = io.TextIOWrapper(csv_file, encoding="utf-8")
                    reader = csv.reader(csv_text)
                    uses_microseconds = date >= datetime(2025, 1, 1, tzinfo=timezone.utc)
                    for row in reader:
                        raw_time = int(row[4])
                        if uses_microseconds:
                            ts = datetime.fromtimestamp(raw_time / 1_000_000, tz=timezone.utc).isoformat()
                        else:
                            ts = datetime.fromtimestamp(raw_time / 1000, tz=timezone.utc).isoformat()
                        trade = {
                            "ts": ts,
                            "price": row[1],
                            "qty": row[2],
                        }
                        trades.append(trade)
            return trades
        except Exception:
            if attempt < max_retries - 1:
                import time
                time.sleep(2 ** attempt)
            else:
                raise
    return []

def main():
    parser = argparse.ArgumentParser(description="Download raw Binance trades for a date range.")
    parser.add_argument("--pair", required=True, help="Trading pair, e.g. BTCUSDT")
    parser.add_argument("--from", dest="from_date", required=True, help="Start date (YYYY-MM-DD), inclusive")
    parser.add_argument("--to", dest="to_date", required=True, help="End date (YYYY-MM-DD), exclusive")
    parser.add_argument("--skip-existing", action="store_true", help="Skip days already downloaded")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    pair = args.pair.upper()
    from_dt = datetime.strptime(args.from_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    to_dt = datetime.strptime(args.to_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    out_dir = Path(f"data/raw/{pair}")
    out_dir.mkdir(parents=True, exist_ok=True)

    current = from_dt
    while current < to_dt:
        out_file = out_dir / f"{current.strftime('%Y-%m-%d')}.csv"
        if out_file.exists():
            if args.force:
                pass  # Overwrite
            elif args.skip_existing:
                print(f"Skipping {out_file} (already exists)")
                current += timedelta(days=1)
                continue
            else:
                print(f"File {out_file} exists. Use --force to overwrite or --skip-existing to skip.")
                current += timedelta(days=1)
                continue
        print(f"Downloading {pair} {current.strftime('%Y-%m-%d')}...")
        trades = download_daily_trades(pair, current)
        total_trades = len(trades)
        filtered = []
        if trades:
            last_price = None
            for t in trades:
                price = float(t["price"])
                if last_price is None:
                    filtered.append(t)
                    last_price = price
                else:
                    if abs(price - last_price) / last_price >= 0.0001:
                        filtered.append(t)
                        last_price = price
            with open(out_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["ts", "price", "qty"])
                writer.writeheader()
                writer.writerows(filtered)
            print(f"Processed {total_trades} trades, saved {len(filtered)} to {out_file}")
        else:
            print(f"No trades found for {pair} {current.strftime('%Y-%m-%d')}")
        current += timedelta(days=1)

if __name__ == "__main__":
    main()
