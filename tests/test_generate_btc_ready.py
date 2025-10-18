import pytest
import pandas as pd
import numpy as np
from decimal import Decimal
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import shutil
import os
from scripts.pipeline.generate_btc_ready import (
    load_trades,
    aggregate_ohlcv,
    add_indicators,
    add_stats,
    calc_max_future_price,
    calc_min_future_price,
)


@pytest.fixture
def sample_trades_df():
    """Creates a sample trades DataFrame for testing"""
    base_time = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    data = {
        # Use datetime objects directly instead of timestamps
        'ts': [(base_time + timedelta(minutes=i)) for i in range(100)],
        'price': [100 + i * 0.5 for i in range(100)],  # Price increases gradually
        'qty': [1.0 + i * 0.01 for i in range(100)]  # Quantity increases gradually
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_raw_data_dir():
    """Creates a temporary directory for raw data"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestLoadTrades:
    """Tests for load_trades function"""
    
    def test_load_trades_basic(self, temp_raw_data_dir, sample_trades_df):
        """Test basic loading of trades from CSV"""
        # Create test CSV file
        pair = "BTCUSDT"
        date = "2020-01-01"
        raw_dir = Path(temp_raw_data_dir) / pair
        raw_dir.mkdir(parents=True)
        csv_path = raw_dir / f"{date}.csv"
        sample_trades_df.to_csv(csv_path, index=False)
        
        # Load trades - load_trades expects (pair, day_path)
        df = load_trades(pair, str(csv_path))
        
        # Verify
        assert len(df) == 100
        assert 'ts' in df.columns
        assert 'price' in df.columns
        assert 'qty' in df.columns
        # ts might be datetime or int64 after parsing
        assert df['price'].dtype == 'float64'
        assert df['qty'].dtype == 'float64'
    
    def test_load_trades_missing_file(self, temp_raw_data_dir):
        """Test loading non-existent file raises error"""
        with pytest.raises(FileNotFoundError):
            load_trades("BTCUSDT", "/nonexistent/path/2020-01-01.csv")


class TestAggregateOHLCV:
    """Tests for aggregate_ohlcv function"""
    
    def test_aggregate_5min(self, sample_trades_df):
        """Test aggregation to 5-minute candles"""
        # aggregate_ohlcv returns a DataFrame with ts as index, not as column
        df = aggregate_ohlcv(sample_trades_df, '5min')
        
        # Verify columns (ts is the index, not a column)
        assert 'open' in df.columns
        assert 'high' in df.columns
        assert 'low' in df.columns
        assert 'close' in df.columns
        assert 'volume' in df.columns
        assert df.index.name == 'ts' or isinstance(df.index, pd.DatetimeIndex)
        
        # Verify data
        assert len(df) > 0
        # For each row, high >= low
        for idx, row in df.iterrows():
            assert row['high'] >= row['low'], f"High ({row['high']}) should be >= Low ({row['low']})"
        
    def test_aggregate_all_windows(self, sample_trades_df):
        """Test aggregation for all time windows"""
        windows = ['5min', '15min', '1h', '4h', '1d']
        
        for window in windows:
            df = aggregate_ohlcv(sample_trades_df, window)
            assert len(df) > 0, f"Failed for window {window}"
            # ts is index, not column
            assert 'open' in df.columns
            assert 'close' in df.columns
    
    def test_aggregate_ohlcv_values(self):
        """Test that OHLCV values are calculated correctly"""
        # Create specific test data with UTC timestamps
        base_time = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        data = {
            'ts': [(base_time + timedelta(minutes=i)) for i in range(10)],
            'price': [100, 105, 110, 95, 90, 100, 105, 110, 115, 120],
            'qty': [1.0] * 10
        }
        df = pd.DataFrame(data)
        
        # Aggregate to 5min
        result = aggregate_ohlcv(df, '5min')
        
        # Should create 2 candles (0-4 minutes, 5-9 minutes)
        assert len(result) == 2
        # First candle: prices [100, 105, 110, 95, 90]
        assert result.iloc[0]['open'] == 100
        assert result.iloc[0]['high'] == 110
        assert result.iloc[0]['low'] == 90
        assert result.iloc[0]['close'] == 90
        # Second candle: prices [100, 105, 110, 115, 120]
        assert result.iloc[1]['open'] == 100
        assert result.iloc[1]['high'] == 120
        assert result.iloc[1]['low'] == 100
        assert result.iloc[1]['close'] == 120


class TestAddIndicators:
    """Tests for add_indicators function"""
    
    def test_add_indicators_columns(self):
        """Test that all indicators are added"""
        # Create test data with enough rows for all indicators
        # We need at least 20 aggregated candles for sma_20
        base_time = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        data = {
            'ts': [(base_time + timedelta(minutes=i)) for i in range(150)],  # More data
            'price': [100 + i * 0.5 for i in range(150)],
            'qty': [1.0] * 150
        }
        trades_df = pd.DataFrame(data)
        df = aggregate_ohlcv(trades_df, '5min')
        df = add_indicators(df)
        
        # Check technical indicators are present - based on actual script
        # With 150 trades at 5min intervals, we should have 30 candles
        expected_indicators = [
            'sma_20', 'ema_20',
            'bb_high', 'bb_low',
        ]
        
        for indicator in expected_indicators:
            assert indicator in df.columns, f"Missing indicator: {indicator}"
    
    def test_add_indicators_with_sufficient_data(self):
        """Test that indicators are added with sufficient data"""
        # Create larger dataset to ensure all indicators
        base_time = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        data = {
            'ts': [(base_time + timedelta(minutes=i)) for i in range(200)],
            'price': [100 + i * 0.1 for i in range(200)],
            'qty': [1.0] * 200
        }
        trades_df = pd.DataFrame(data)
        df = aggregate_ohlcv(trades_df, '5min')
        df = add_indicators(df)
        
        # With 200 trades, we should have ~40 candles, enough for all indicators
        if len(df) >= 35:
            assert 'macd' in df.columns
            assert 'macd_signal' in df.columns
        if len(df) >= 27:
            assert 'adx_14' in df.columns
        if len(df) >= 15:
            assert 'rsi_14' in df.columns
            assert 'atr_14' in df.columns
            assert 'stoch_k' in df.columns
            assert 'stoch_d' in df.columns
        if len(df) >= 20:
            assert 'sma_20' in df.columns
            assert 'ema_20' in df.columns


class TestAddStats:
    """Tests for add_stats function"""
    
    def test_add_stats_columns(self):
        """Test that all statistics are added"""
        base_time = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        data = {
            'ts': [(base_time + timedelta(minutes=i)) for i in range(100)],
            'price': [100 + i * 0.5 for i in range(100)],
            'qty': [1.0] * 100
        }
        trades_df = pd.DataFrame(data)
        df = aggregate_ohlcv(trades_df, '5min')
        df = add_indicators(df)
        df = add_stats(df)
        
        # Based on actual script - uses 'return' not 'returns', etc.
        expected_stats = [
            'return',  # not 'returns'
            'volatility',  # not 'volatility_10'/'volatility_20'
            'drawdown',
            'sharpe',  # not 'sharpe_ratio'
            'sortino',  # not 'sortino_ratio'
            'skew',  # not 'skewness'
            'kurtosis'
        ]
        
        for stat in expected_stats:
            assert stat in df.columns, f"Missing statistic: {stat}"
    
    def test_add_stats_returns_calculation(self):
        """Test that returns are calculated correctly"""
        data = {
            'ts': [datetime(2020, 1, 1, 0, i, 0, tzinfo=timezone.utc) for i in range(5)],
            'open': [100, 110, 105, 115, 120],
            'high': [110, 115, 115, 120, 125],
            'low': [95, 105, 100, 110, 115],
            'close': [110, 105, 115, 120, 125],
            'volume': [1.0] * 5
        }
        df = pd.DataFrame(data)
        df = df.set_index('ts')
        df = add_stats(df)
        
        # First return should be NaN (no previous value)
        assert pd.isna(df.iloc[0]['return'])
        
        # Second return: (105 - 110) / 110 = -0.04545... (more precision)
        assert pytest.approx(df.iloc[1]['return'], abs=0.0001) == -0.0454545
        
        # Third return: (115 - 105) / 105 = 0.0952... (more precision)
        assert pytest.approx(df.iloc[2]['return'], abs=0.0001) == 0.0952381


class TestCalculateFuturePrices:
    """Tests for future price calculation functions"""
    
    def test_calc_max_future_price_integration(self):
        """Test max future price calculation with realistic data"""
        # Create data where price goes up without triggering STEP_BACK or LOSS
        prices = [100, 101, 102, 103, 104, 105]  # Gradual increase
        timestamps = [i * 1000 for i in range(len(prices))]
        
        max_price, max_perc, max_date = calc_max_future_price(prices, timestamps)
        
        assert len(max_price) == len(prices)
        assert len(max_perc) == len(prices)
        assert len(max_date) == len(prices)
        
        # First trade should see max price of 105 (no big drop to trigger STEP_BACK)
        assert max_price[0] == 105
        assert max_date[0] == timestamps[5]
    
    def test_calc_min_future_price_integration(self):
        """Test min future price calculation with realistic data"""
        # Create data where price goes down without triggering STEP_BACK or LOSS
        prices = [100, 99, 98, 97, 96, 95]  # Gradual decrease
        timestamps = [i * 1000 for i in range(len(prices))]
        
        min_price, min_perc, min_date = calc_min_future_price(prices, timestamps)
        
        assert len(min_price) == len(prices)
        assert len(min_perc) == len(prices)
        assert len(min_date) == len(prices)
        
        # First trade should see min price of 95 (no big jump to trigger STEP_BACK)
        assert min_price[0] == 95
        assert min_date[0] == timestamps[5]


class TestPipelineIntegration:
    """Integration tests for the complete pipeline"""
    
    def test_full_pipeline_basic(self, temp_raw_data_dir, sample_trades_df):
        """Test complete pipeline from raw trades to ready data"""
        # Create test CSV file
        pair = "BTCUSDT"
        date = "2020-01-01"
        raw_dir = Path(temp_raw_data_dir) / pair
        raw_dir.mkdir(parents=True)
        csv_path = raw_dir / f"{date}.csv"
        sample_trades_df.to_csv(csv_path, index=False)
        
        # Execute pipeline steps - load_trades expects (pair, day_path)
        trades_df = load_trades(pair, str(csv_path))
        
        # Process all windows and merge
        windows = ['5min', '15min', '1h', '4h', '1d']
        merged_dfs = []
        
        for window in windows:
            df = aggregate_ohlcv(trades_df.copy(), window)
            df = add_indicators(df)
            df = add_stats(df)
            
            # Reset index to convert ts from index to column
            df = df.reset_index()
            
            # Rename columns with window prefix
            rename_dict = {col: f"{window}_{col}" for col in df.columns if col != 'ts'}
            df = df.rename(columns=rename_dict)
            merged_dfs.append(df)
        
        # Merge all windows
        merged = merged_dfs[0]
        for df in merged_dfs[1:]:
            merged = merged.merge(df, on='ts', how='outer')
        
        # Verify merged result
        assert 'ts' in merged.columns
        assert len(merged) > 0
        
        # Check that each window has its columns
        for window in windows:
            assert f"{window}_open" in merged.columns
            assert f"{window}_close" in merged.columns
    
    def test_column_naming_consistency(self, sample_trades_df):
        """Test that column naming is consistent across windows"""
        windows = ['5min', '15min', '1h']
        
        for window in windows:
            df = aggregate_ohlcv(sample_trades_df.copy(), window)
            df = add_indicators(df)
            df = add_stats(df)
            
            # Check base columns exist before renaming (ts is index)
            assert 'open' in df.columns
            # 'return' should exist from add_stats
            assert 'return' in df.columns
            
            # Reset index to make ts a column
            df = df.reset_index()
            assert 'ts' in df.columns
            
            # Rename with window prefix
            rename_dict = {col: f"{window}_{col}" for col in df.columns if col != 'ts'}
            df = df.rename(columns=rename_dict)
            
            # Check renamed columns
            assert f"{window}_open" in df.columns
            assert f"{window}_return" in df.columns
            assert 'ts' in df.columns  # ts should not be renamed


class TestDataQuality:
    """Tests for data quality and edge cases"""
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame"""
        df = pd.DataFrame(columns=['ts', 'price', 'qty'])
        
        # aggregate_ohlcv should handle empty data gracefully
        result = aggregate_ohlcv(df, '5min')
        assert len(result) == 0
    
    def test_single_trade(self):
        """Test handling of single trade"""
        data = {
            'ts': [1577836800000],
            'price': [100.0],
            'qty': [1.0]
        }
        df = pd.DataFrame(data)
        
        result = aggregate_ohlcv(df, '5min')
        assert len(result) == 1
        assert result.iloc[0]['open'] == 100.0
        assert result.iloc[0]['close'] == 100.0
        assert result.iloc[0]['high'] == 100.0
        assert result.iloc[0]['low'] == 100.0
    
    def test_nan_handling_in_indicators(self):
        """Test that NaN values are handled properly in indicators"""
        # Create data with enough points for indicators
        base_time = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        data = {
            'ts': [(base_time + timedelta(minutes=i)) for i in range(100)],
            'price': [100 + i for i in range(100)],
            'qty': [1.0] * 100,
        }
        df = pd.DataFrame(data)
        
        ohlcv = aggregate_ohlcv(df, '5min')
        result = add_indicators(ohlcv)
        
        # Indicators should not crash on calculation
        # Check if at least some indicator columns exist (depending on data length)
        if len(result) >= 20:
            assert 'sma_20' in result.columns
        
        # Not all values should be NaN (some might be at beginning due to warmup)
        if len(result) >= 20:
            assert not result['sma_20'].isna().all()
