import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import shutil
from scripts.pipeline.generate_pair_ready import (
    load_btc_ready,
    merge_with_btc_data,
    process_day,
)


@pytest.fixture
def temp_data_dirs():
    """Creates temporary directories for test data"""
    temp_base = tempfile.mkdtemp()
    raw_dir = Path(temp_base) / "raw"
    btc_ready_dir = Path(temp_base) / "ready" / "BTCUSDT"
    output_dir = Path(temp_base) / "ready"
    
    raw_dir.mkdir(parents=True)
    btc_ready_dir.mkdir(parents=True)
    
    yield {
        "base": temp_base,
        "raw": str(raw_dir),
        "btc_ready": str(btc_ready_dir),
        "output": str(output_dir)
    }
    
    shutil.rmtree(temp_base)


@pytest.fixture
def sample_btc_ready_df():
    """Creates a sample BTC ready DataFrame"""
    base_time = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    data = {
        'ts': [(base_time + timedelta(minutes=i*5)) for i in range(20)],
        '5min_open': [7000 + i*10 for i in range(20)],
        '5min_high': [7010 + i*10 for i in range(20)],
        '5min_low': [6990 + i*10 for i in range(20)],
        '5min_close': [7005 + i*10 for i in range(20)],
        '5min_volume': [100.0 + i for i in range(20)],
        'max_future_price': [7200 + i*5 for i in range(20)],
        'max_future_price_perc': [0.02 + i*0.001 for i in range(20)],
        'max_future_price_date': [(base_time + timedelta(hours=1)) for _ in range(20)],
        'min_future_price': [6900 - i*5 for i in range(20)],
        'min_future_price_perc': [0.01 + i*0.001 for i in range(20)],
        'min_future_price_date': [(base_time + timedelta(hours=2)) for _ in range(20)],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_pair_ready_df():
    """Creates a sample pair ready DataFrame"""
    base_time = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    data = {
        'ts': [(base_time + timedelta(minutes=i*5)) for i in range(10)],
        '5min_open': [130 + i for i in range(10)],
        '5min_high': [135 + i for i in range(10)],
        '5min_low': [125 + i for i in range(10)],
        '5min_close': [132 + i for i in range(10)],
        '5min_volume': [1000.0 + i*10 for i in range(10)],
    }
    return pd.DataFrame(data)


class TestLoadBtcReady:
    """Tests for load_btc_ready function"""
    
    def test_load_btc_ready_basic(self, temp_data_dirs, sample_btc_ready_df):
        """Test basic loading of BTC ready file"""
        # Create test CSV file
        btc_ready_path = Path(temp_data_dirs["btc_ready"]) / "2020-01-01.csv"
        sample_btc_ready_df.to_csv(btc_ready_path, index=False)
        
        # Load BTC ready data
        df = load_btc_ready(temp_data_dirs["btc_ready"], "2020-01-01")
        
        # Verify
        assert len(df) == 20
        assert 'ts' in df.columns
        assert '5min_close' in df.columns
        assert 'max_future_price' in df.columns
        # Verify ts is timezone-aware
        assert df['ts'].dt.tz is not None
    
    def test_load_btc_ready_missing_file(self, temp_data_dirs):
        """Test loading non-existent BTC ready file raises error"""
        with pytest.raises(FileNotFoundError):
            load_btc_ready(temp_data_dirs["btc_ready"], "2020-01-01")


class TestMergeWithBtcData:
    """Tests for merge_with_btc_data function"""
    
    def test_merge_basic(self, sample_pair_ready_df, sample_btc_ready_df):
        """Test basic merging of pair data with BTC data"""
        windows = ['5min']
        
        result = merge_with_btc_data(
            sample_pair_ready_df.copy(),
            sample_btc_ready_df.copy(),
            windows
        )
        
        # Verify BTC columns were added
        assert 'btc_5min_open' in result.columns
        assert 'btc_5min_close' in result.columns
        assert 'btc_5min_high' in result.columns
        assert 'btc_5min_low' in result.columns
        assert 'btc_5min_volume' in result.columns
        
        # Verify future price columns were added
        assert 'btc_max_future_price' in result.columns
        assert 'btc_max_future_price_perc' in result.columns
        assert 'btc_min_future_price' in result.columns
        assert 'btc_min_future_price_perc' in result.columns
        
        # Verify original columns are preserved
        assert '5min_open' in result.columns
        assert '5min_close' in result.columns
        assert len(result) == len(sample_pair_ready_df)
    
    def test_merge_closest_timestamp(self):
        """Test that closest BTC timestamp is selected correctly"""
        base_time = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        
        # BTC data every 5 minutes
        btc_df = pd.DataFrame({
            'ts': [(base_time + timedelta(minutes=i*5)) for i in range(5)],
            '5min_close': [7000, 7010, 7020, 7030, 7040],
        })
        
        # Pair data at odd times (should match to previous BTC timestamp)
        pair_df = pd.DataFrame({
            'ts': [
                base_time + timedelta(minutes=3),  # Should match BTC at 0 min (7000)
                base_time + timedelta(minutes=7),  # Should match BTC at 5 min (7010)
                base_time + timedelta(minutes=12), # Should match BTC at 10 min (7020)
            ],
            '5min_close': [130, 131, 132],
        })
        
        result = merge_with_btc_data(pair_df, btc_df, ['5min'])
        
        # Verify correct BTC values were matched
        assert result.iloc[0]['btc_5min_close'] == 7000
        assert result.iloc[1]['btc_5min_close'] == 7010
        assert result.iloc[2]['btc_5min_close'] == 7020
    
    def test_merge_no_future_btc_data(self):
        """Test merging when no BTC data exists for future timestamps"""
        base_time = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        
        # BTC data only up to 10 minutes
        btc_df = pd.DataFrame({
            'ts': [(base_time + timedelta(minutes=i*5)) for i in range(3)],
            '5min_close': [7000, 7010, 7020],
        })
        
        # Pair data extends beyond BTC data
        pair_df = pd.DataFrame({
            'ts': [
                base_time + timedelta(minutes=5),  # Has BTC match
                base_time + timedelta(minutes=15), # Has BTC match (uses last available)
                base_time + timedelta(minutes=25), # No newer BTC, uses last (7020)
            ],
            '5min_close': [130, 131, 132],
        })
        
        result = merge_with_btc_data(pair_df, btc_df, ['5min'])
        
        # All should use the most recent available BTC data
        assert result.iloc[0]['btc_5min_close'] == 7010
        assert result.iloc[1]['btc_5min_close'] == 7020
        assert result.iloc[2]['btc_5min_close'] == 7020
    
    def test_merge_multiple_windows(self, sample_pair_ready_df, sample_btc_ready_df):
        """Test merging with multiple time windows"""
        # Add 15min data to both dataframes
        sample_btc_ready_df['15min_close'] = sample_btc_ready_df['5min_close'] + 5
        sample_pair_ready_df['15min_close'] = sample_pair_ready_df['5min_close'] + 2
        
        windows = ['5min', '15min']
        result = merge_with_btc_data(
            sample_pair_ready_df.copy(),
            sample_btc_ready_df.copy(),
            windows
        )
        
        # Verify both window columns were added
        assert 'btc_5min_close' in result.columns
        assert 'btc_15min_close' in result.columns
    
    def test_merge_timezone_handling(self):
        """Test that timezone-naive timestamps are handled correctly"""
        base_time = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        
        # BTC data with timezone
        btc_df = pd.DataFrame({
            'ts': [(base_time + timedelta(minutes=i*5)) for i in range(3)],
            '5min_close': [7000, 7010, 7020],
        })
        
        # Pair data without timezone (naive)
        pair_df = pd.DataFrame({
            'ts': pd.to_datetime([
                '2020-01-01 00:00:00',
                '2020-01-01 00:05:00',
                '2020-01-01 00:10:00',
            ]),
            '5min_close': [130, 131, 132],
        })
        
        # Should not raise TypeError
        result = merge_with_btc_data(pair_df, btc_df, ['5min'])
        
        # Verify merge completed successfully
        assert 'btc_5min_close' in result.columns
        assert len(result) == 3


class TestProcessDay:
    """Tests for process_day function"""
    
    def test_process_day_creates_output(self, temp_data_dirs):
        """Test that process_day creates output file"""
        # Create raw trade data for ETHUSDT
        pair = "ETHUSDT"
        date = "2020-01-01"
        
        raw_pair_dir = Path(temp_data_dirs["raw"]) / pair
        raw_pair_dir.mkdir(parents=True)
        
        base_time = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        trades_data = {
            'ts': [(base_time + timedelta(minutes=i)) for i in range(100)],
            'price': [130 + i * 0.1 for i in range(100)],
            'qty': [1.0 + i * 0.01 for i in range(100)]
        }
        trades_df = pd.DataFrame(trades_data)
        trades_df.to_csv(raw_pair_dir / f"{date}.csv", index=False)
        
        # Create BTC ready data
        btc_data = {
            'ts': [(base_time + timedelta(minutes=i*5)) for i in range(20)],
            '5min_open': [7000 + i*10 for i in range(20)],
            '5min_close': [7005 + i*10 for i in range(20)],
            '5min_high': [7010 + i*10 for i in range(20)],
            '5min_low': [6990 + i*10 for i in range(20)],
            '5min_volume': [100.0 for _ in range(20)],
            'max_future_price': [7200 for _ in range(20)],
            'max_future_price_perc': [0.02 for _ in range(20)],
            'max_future_price_date': [(base_time + timedelta(hours=1)) for _ in range(20)],
            'min_future_price': [6900 for _ in range(20)],
            'min_future_price_perc': [0.01 for _ in range(20)],
            'min_future_price_date': [(base_time + timedelta(hours=2)) for _ in range(20)],
        }
        btc_df = pd.DataFrame(btc_data)
        btc_df.to_csv(Path(temp_data_dirs["btc_ready"]) / f"{date}.csv", index=False)
        
        # Process the day
        process_day(
            pair=pair,
            date=date,
            raw_dir=temp_data_dirs["raw"],
            btc_ready_dir=temp_data_dirs["btc_ready"],
            output_dir=temp_data_dirs["output"],
            windows=['5min'],
            overwrite=True
        )
        
        # Verify output file was created
        output_file = Path(temp_data_dirs["output"]) / pair / f"{date}.csv"
        assert output_file.exists()
        
        # Load and verify output
        result_df = pd.read_csv(output_file)
        assert len(result_df) > 0
        assert 'ts' in result_df.columns
        assert '5min_close' in result_df.columns
        assert 'btc_5min_close' in result_df.columns
        assert 'max_future_price' in result_df.columns
        assert 'btc_max_future_price' in result_df.columns


class TestEdgeCases:
    """Tests for edge cases and error handling"""
    
    def test_empty_pair_dataframe(self, sample_btc_ready_df):
        """Test handling of empty pair DataFrame"""
        empty_df = pd.DataFrame(columns=['ts', '5min_close'])
        
        result = merge_with_btc_data(empty_df, sample_btc_ready_df, ['5min'])
        
        # Should return empty DataFrame with BTC columns added
        assert len(result) == 0
        assert 'btc_5min_close' in result.columns
    
    def test_missing_btc_columns(self, sample_pair_ready_df):
        """Test handling when BTC data is missing expected columns"""
        # BTC data with minimal columns
        base_time = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        btc_df = pd.DataFrame({
            'ts': [(base_time + timedelta(minutes=i*5)) for i in range(10)],
            '5min_close': [7000 + i*10 for i in range(10)],
            # Missing other 5min columns
        })
        
        result = merge_with_btc_data(sample_pair_ready_df, btc_df, ['5min'])
        
        # Should only add columns that exist in BTC
        assert 'btc_5min_close' in result.columns
        assert 'btc_5min_open' not in result.columns  # Wasn't in BTC data
