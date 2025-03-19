"""
Technical Indicators

This module calculates various technical indicators for time series data.
"""

import pandas as pd
import numpy as np
import logging
import warnings

logger = logging.getLogger('marketmind.preprocessing.technical_indicators')

class TechnicalIndicator:
    """
    Calculates technical indicators for financial time series data.
    """
    
    def __init__(self, config=None):
        """
        Initialize the TechnicalIndicator calculator with configuration.
        
        Args:
            config (dict): Configuration dictionary containing indicator settings.
        """
        from .. import load_config
        self.config = config or load_config()
        self.indicators_config = self.config.get('preprocessing', {}).get('technical_indicators', [])
        
        # Filter warnings from pandas operations
        warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    def add_all_indicators(self, df, inplace=False):
        """
        Add all configured technical indicators to the DataFrame.
        
        Args:
            df (pandas.DataFrame): DataFrame with OHLC price data
            inplace (bool): Whether to modify the DataFrame in place
            
        Returns:
            pandas.DataFrame: DataFrame with added technical indicators
        """
        if not inplace:
            df = df.copy()
        
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            # Try to find alternative column names
            alt_names = {
                'open': ['Open'],
                'high': ['High'],
                'low': ['Low'],
                'close': ['Close', 'Adj Close'],
                'volume': ['Volume']
            }
            
            for missing_col in missing_columns:
                for alt_name in alt_names.get(missing_col, []):
                    if alt_name in df.columns:
                        df[missing_col] = df[alt_name]
                        logger.info(f"Using '{alt_name}' column as '{missing_col}'")
                        break
        
        # Check if we still have missing columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}. Some indicators may not be calculated correctly.")
        
        # Add indicators based on configuration
        for indicator_config in self.indicators_config:
            indicator_name = indicator_config.get('name', '').upper()
            
            if indicator_name == 'SMA':
                periods = indicator_config.get('periods', [20])
                for period in periods:
                    self.add_sma(df, period, inplace=True)
                    
            elif indicator_name == 'EMA':
                periods = indicator_config.get('periods', [20])
                for period in periods:
                    self.add_ema(df, period, inplace=True)
                    
            elif indicator_name == 'RSI':
                period = indicator_config.get('period', 14)
                self.add_rsi(df, period, inplace=True)
                
            elif indicator_name == 'MACD':
                fast_period = indicator_config.get('fast_period', 12)
                slow_period = indicator_config.get('slow_period', 26)
                signal_period = indicator_config.get('signal_period', 9)
                self.add_macd(df, fast_period, slow_period, signal_period, inplace=True)
                
            elif indicator_name == 'BBANDS':
                period = indicator_config.get('period', 20)
                std_dev = indicator_config.get('std_dev', 2)
                self.add_bollinger_bands(df, period, std_dev, inplace=True)
                
            else:
                logger.warning(f"Unknown indicator: {indicator_name}")
        
        return df
    
    def add_sma(self, df, period=20, column='close', inplace=False):
        """
        Add Simple Moving Average (SMA) to the DataFrame.
        
        Args:
            df (pandas.DataFrame): DataFrame with price data
            period (int): Period for SMA calculation
            column (str): Column to use for calculation
            inplace (bool): Whether to modify the DataFrame in place
            
        Returns:
            pandas.DataFrame: DataFrame with added SMA
        """
        if not inplace:
            df = df.copy()
        
        if column not in df.columns:
            logger.warning(f"Column '{column}' not found in DataFrame. SMA not calculated.")
            return df
        
        col_name = f'sma_{period}'
        df[col_name] = df[column].rolling(window=period).mean()
        
        return df
    
    def add_ema(self, df, period=20, column='close', inplace=False):
        """
        Add Exponential Moving Average (EMA) to the DataFrame.
        
        Args:
            df (pandas.DataFrame): DataFrame with price data
            period (int): Period for EMA calculation
            column (str): Column to use for calculation
            inplace (bool): Whether to modify the DataFrame in place
            
        Returns:
            pandas.DataFrame: DataFrame with added EMA
        """
        if not inplace:
            df = df.copy()
        
        if column not in df.columns:
            logger.warning(f"Column '{column}' not found in DataFrame. EMA not calculated.")
            return df
        
        col_name = f'ema_{period}'
        df[col_name] = df[column].ewm(span=period, adjust=False).mean()
        
        return df
    
    def add_rsi(self, df, period=14, column='close', inplace=False):
        """
        Add Relative Strength Index (RSI) to the DataFrame.
        
        Args:
            df (pandas.DataFrame): DataFrame with price data
            period (int): Period for RSI calculation
            column (str): Column to use for calculation
            inplace (bool): Whether to modify the DataFrame in place
            
        Returns:
            pandas.DataFrame: DataFrame with added RSI
        """
        if not inplace:
            df = df.copy()
        
        if column not in df.columns:
            logger.warning(f"Column '{column}' not found in DataFrame. RSI not calculated.")
            return df
        
        col_name = f'rsi_{period}'
        
        # Calculate price changes
        delta = df[column].diff()
        
        # Get gains and losses
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        df[col_name] = 100 - (100 / (1 + rs))
        
        return df
    
    def add_macd(self, df, fast_period=12, slow_period=26, signal_period=9, column='close', inplace=False):
        """
        Add Moving Average Convergence Divergence (MACD) to the DataFrame.
        
        Args:
            df (pandas.DataFrame): DataFrame with price data
            fast_period (int): Fast period for EMA calculation
            slow_period (int): Slow period for EMA calculation
            signal_period (int): Signal period for EMA calculation
            column (str): Column to use for calculation
            inplace (bool): Whether to modify the DataFrame in place
            
        Returns:
            pandas.DataFrame: DataFrame with added MACD
        """
        if not inplace:
            df = df.copy()
        
        if column not in df.columns:
            logger.warning(f"Column '{column}' not found in DataFrame. MACD not calculated.")
            return df
        
        # Calculate fast and slow EMAs
        ema_fast = df[column].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df[column].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        df['macd_line'] = ema_fast - ema_slow
        
        # Calculate signal line
        df['macd_signal'] = df['macd_line'].ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        df['macd_histogram'] = df['macd_line'] - df['macd_signal']
        
        return df
    
    def add_bollinger_bands(self, df, period=20, std_dev=2, column='close', inplace=False):
        """
        Add Bollinger Bands to the DataFrame.
        
        Args:
            df (pandas.DataFrame): DataFrame with price data
            period (int): Period for moving average calculation
            std_dev (float): Number of standard deviations for bands
            column (str): Column to use for calculation
            inplace (bool): Whether to modify the DataFrame in place
            
        Returns:
            pandas.DataFrame: DataFrame with added Bollinger Bands
        """
        if not inplace:
            df = df.copy()
        
        if column not in df.columns:
            logger.warning(f"Column '{column}' not found in DataFrame. Bollinger Bands not calculated.")
            return df
        
        # Calculate middle band (SMA)
        df['bb_middle'] = df[column].rolling(window=period).mean()
        
        # Calculate standard deviation
        rolling_std = df[column].rolling(window=period).std()
        
        # Calculate upper and lower bands
        df['bb_upper'] = df['bb_middle'] + (rolling_std * std_dev)
        df['bb_lower'] = df['bb_middle'] - (rolling_std * std_dev)
        
        # Calculate bandwidth
        df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Calculate percent B
        df['bb_percent_b'] = (df[column] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    def add_atr(self, df, period=14, inplace=False):
        """
        Add Average True Range (ATR) to the DataFrame.
        
        Args:
            df (pandas.DataFrame): DataFrame with price data
            period (int): Period for ATR calculation
            inplace (bool): Whether to modify the DataFrame in place
            
        Returns:
            pandas.DataFrame: DataFrame with added ATR
        """
        if not inplace:
            df = df.copy()
        
        required_columns = ['high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}. ATR not calculated.")
            return df
        
        # Calculate true range
        df['tr1'] = abs(df['high'] - df['low'])
        df['tr2'] = abs(df['high'] - df['close'].shift())
        df['tr3'] = abs(df['low'] - df['close'].shift())
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate ATR
        df[f'atr_{period}'] = df['true_range'].rolling(window=period).mean()
        
        # Drop intermediate columns
        df.drop(['tr1', 'tr2', 'tr3', 'true_range'], axis=1, inplace=True)
        
        return df
    
    def add_adx(self, df, period=14, inplace=False):
        """
        Add Average Directional Index (ADX) to the DataFrame.
        
        Args:
            df (pandas.DataFrame): DataFrame with price data
            period (int): Period for ADX calculation
            inplace (bool): Whether to modify the DataFrame in place
            
        Returns:
            pandas.DataFrame: DataFrame with added ADX
        """
        if not inplace:
            df = df.copy()
        
        required_columns = ['high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}. ADX not calculated.")
            return df
        
        # Calculate +DM and -DM
        df['up_move'] = df['high'].diff()
        df['down_move'] = df['low'].diff(-1).abs()
        
        df['plus_dm'] = 0
        df['minus_dm'] = 0
        
        # Calculate +DM
        dm_plus_condition = (df['up_move'] > df['down_move']) & (df['up_move'] > 0)
        df.loc[dm_plus_condition, 'plus_dm'] = df.loc[dm_plus_condition, 'up_move']
        
        # Calculate -DM
        dm_minus_condition = (df['down_move'] > df['up_move']) & (df['down_move'] > 0)
        df.loc[dm_minus_condition, 'minus_dm'] = df.loc[dm_minus_condition, 'down_move']
        
        # Add ATR
        self.add_atr(df, period, inplace=True)
        
        # Calculate smoothed +DM and -DM
        atr_col = f'atr_{period}'
        df['plus_di'] = 100 * (df['plus_dm'].rolling(window=period).mean() / df[atr_col])
        df['minus_di'] = 100 * (df['minus_dm'].rolling(window=period).mean() / df[atr_col])
        
        # Calculate directional index
        df['dx'] = 100 * (abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']))
        
        # Calculate ADX
        df[f'adx_{period}'] = df['dx'].rolling(window=period).mean()
        
        # Drop intermediate columns
        df.drop(['up_move', 'down_move', 'plus_dm', 'minus_dm', 'dx'], axis=1, inplace=True)
        
        return df
    
    def add_stochastic(self, df, k_period=14, d_period=3, inplace=False):
        """
        Add Stochastic Oscillator to the DataFrame.
        
        Args:
            df (pandas.DataFrame): DataFrame with price data
            k_period (int): Period for %K calculation
            d_period (int): Period for %D calculation
            inplace (bool): Whether to modify the DataFrame in place
            
        Returns:
            pandas.DataFrame: DataFrame with added Stochastic Oscillator
        """
        if not inplace:
            df = df.copy()
        
        required_columns = ['high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}. Stochastic Oscillator not calculated.")
            return df
        
        # Calculate %K
        df['stoch_highest_high'] = df['high'].rolling(window=k_period).max()
        df['stoch_lowest_low'] = df['low'].rolling(window=k_period).min()
        df['stoch_k'] = 100 * ((df['close'] - df['stoch_lowest_low']) / 
                              (df['stoch_highest_high'] - df['stoch_lowest_low']))
        
        # Calculate %D
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
        
        # Drop intermediate columns
        df.drop(['stoch_highest_high', 'stoch_lowest_low'], axis=1, inplace=True)
        
        return df
    
    def add_obv(self, df, inplace=False):
        """
        Add On-Balance Volume (OBV) to the DataFrame.
        
        Args:
            df (pandas.DataFrame): DataFrame with price and volume data
            inplace (bool): Whether to modify the DataFrame in place
            
        Returns:
            pandas.DataFrame: DataFrame with added OBV
        """
        if not inplace:
            df = df.copy()
        
        required_columns = ['close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}. OBV not calculated.")
            return df
        
        # Calculate OBV
        df['obv'] = 0
        
        # First value is just the volume
        df.loc[df.index[0], 'obv'] = df.loc[df.index[0], 'volume']
        
        # Calculate based on price movement
        for i in range(1, len(df)):
            if df.loc[df.index[i], 'close'] > df.loc[df.index[i-1], 'close']:
                df.loc[df.index[i], 'obv'] = df.loc[df.index[i-1], 'obv'] + df.loc[df.index[i], 'volume']
            elif df.loc[df.index[i], 'close'] < df.loc[df.index[i-1], 'close']:
                df.loc[df.index[i], 'obv'] = df.loc[df.index[i-1], 'obv'] - df.loc[df.index[i], 'volume']
            else:
                df.loc[df.index[i], 'obv'] = df.loc[df.index[i-1], 'obv']
        
        return df
    
    def add_ichimoku(self, df, tenkan_period=9, kijun_period=26, senkou_span_b_period=52, inplace=False):
        """
        Add Ichimoku Cloud to the DataFrame.
        
        Args:
            df (pandas.DataFrame): DataFrame with price data
            tenkan_period (int): Period for Tenkan-sen calculation
            kijun_period (int): Period for Kijun-sen calculation
            senkou_span_b_period (int): Period for Senkou Span B calculation
            inplace (bool): Whether to modify the DataFrame in place
            
        Returns:
            pandas.DataFrame: DataFrame with added Ichimoku Cloud
        """
        if not inplace:
            df = df.copy()
        
        required_columns = ['high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}. Ichimoku Cloud not calculated.")
            return df
        
        # Calculate Tenkan-sen (Conversion Line)
        period_high = df['high'].rolling(window=tenkan_period).max()
        period_low = df['low'].rolling(window=tenkan_period).min()
        df['ichimoku_tenkan_sen'] = (period_high + period_low) / 2
        
        # Calculate Kijun-sen (Base Line)
        period_high = df['high'].rolling(window=kijun_period).max()
        period_low = df['low'].rolling(window=kijun_period).min()
        df['ichimoku_kijun_sen'] = (period_high + period_low) / 2
        
        # Calculate Senkou Span A (Leading Span A)
        df['ichimoku_senkou_span_a'] = ((df['ichimoku_tenkan_sen'] + df['ichimoku_kijun_sen']) / 2).shift(kijun_period)
        
        # Calculate Senkou Span B (Leading Span B)
        period_high = df['high'].rolling(window=senkou_span_b_period).max()
        period_low = df['low'].rolling(window=senkou_span_b_period).min()
        df['ichimoku_senkou_span_b'] = ((period_high + period_low) / 2).shift(kijun_period)
        
        # Calculate Chikou Span (Lagging Span)
        df['ichimoku_chikou_span'] = df['close'].shift(-kijun_period)
        
        return df 