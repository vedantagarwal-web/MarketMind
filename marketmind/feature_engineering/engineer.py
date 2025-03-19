"""
Feature Engineering Module

This module handles feature engineering for market data analysis.
"""

import pandas as pd
import numpy as np
import logging
import talib
from datetime import datetime, timedelta
from scipy import stats
from ..utils.database import DatabaseConnector

# Initialize logger
logger = logging.getLogger('marketmind.feature_engineering.engineer')

class FeatureEngineer:
    """
    Feature engineering for market data analysis.
    """
    
    def __init__(self, config=None):
        """
        Initialize the FeatureEngineer with configuration.
        
        Args:
            config (dict): Configuration dictionary containing feature engineering settings.
        """
        from .. import load_config
        self.config = config or load_config()
        self.db_connector = DatabaseConnector(self.config)
        
        # Configure feature sets
        self.price_features = self.config.get('feature_engineering', {}).get('price_features', [
            'sma', 'ema', 'rsi', 'macd', 'bbands', 'atr'
        ])
        self.volume_features = self.config.get('feature_engineering', {}).get('volume_features', [
            'vwap', 'obv', 'ad'
        ])
        self.return_features = self.config.get('feature_engineering', {}).get('return_features', [
            'daily_return', 'log_return', 'volatility'
        ])
        self.lagged_features = self.config.get('feature_engineering', {}).get('lagged_features', True)
        self.lag_periods = self.config.get('feature_engineering', {}).get('lag_periods', [1, 5, 10, 20])
    
    def create_features(self, df, include_price=True, include_volume=True, 
                        include_returns=True, include_lagged=True):
        """
        Create technical indicators and derived features from stock data.
        
        Args:
            df (pandas.DataFrame): DataFrame containing OHLCV data
            include_price (bool): Whether to include price-based features
            include_volume (bool): Whether to include volume-based features
            include_returns (bool): Whether to include return-based features
            include_lagged (bool): Whether to include lagged features
            
        Returns:
            pandas.DataFrame: DataFrame with added features
        """
        if df is None or df.empty:
            logger.warning("No data provided for feature engineering")
            return pd.DataFrame()
        
        # Make a copy of the input dataframe to avoid modifying the original
        df = df.copy()
        
        # Ensure DataFrame has expected columns
        required_cols = ['open', 'high', 'low', 'close']
        if include_volume:
            required_cols.append('volume')
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return df
        
        # Convert column names to lowercase for consistency
        df.columns = [col.lower() for col in df.columns]
        
        # Ensure 'date' is not in the index if it's a column
        if 'date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        # Create features
        if include_price:
            df = self._create_price_features(df)
        
        if include_volume and 'volume' in df.columns:
            df = self._create_volume_features(df)
        
        if include_returns:
            df = self._create_return_features(df)
        
        if include_lagged and self.lagged_features:
            df = self._create_lagged_features(df)
        
        # Drop NaN values that might have been introduced by feature creation
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Log the number of features created
        feature_count = len(df.columns)
        logger.info(f"Created {feature_count} features (including original columns)")
        
        return df
    
    def _create_price_features(self, df):
        """
        Create price-based technical indicators.
        
        Args:
            df (pandas.DataFrame): DataFrame containing OHLCV data
            
        Returns:
            pandas.DataFrame: DataFrame with added price-based features
        """
        logger.info("Creating price-based features")
        
        # Extract columns for easier use
        open_prices = df['open'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        close_prices = df['close'].values
        
        for feature in self.price_features:
            try:
                if feature == 'sma':
                    # Simple Moving Averages
                    periods = [5, 10, 20, 50, 200]
                    for period in periods:
                        df[f'sma_{period}'] = talib.SMA(close_prices, timeperiod=period)
                        # Add price distance from SMA
                        df[f'dist_sma_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']
                
                elif feature == 'ema':
                    # Exponential Moving Averages
                    periods = [5, 10, 20, 50, 200]
                    for period in periods:
                        df[f'ema_{period}'] = talib.EMA(close_prices, timeperiod=period)
                        # Add price distance from EMA
                        df[f'dist_ema_{period}'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}']
                
                elif feature == 'rsi':
                    # Relative Strength Index
                    periods = [7, 14, 21]
                    for period in periods:
                        df[f'rsi_{period}'] = talib.RSI(close_prices, timeperiod=period)
                
                elif feature == 'macd':
                    # Moving Average Convergence Divergence
                    macd, macd_signal, macd_hist = talib.MACD(
                        close_prices, fastperiod=12, slowperiod=26, signalperiod=9
                    )
                    df['macd'] = macd
                    df['macd_signal'] = macd_signal
                    df['macd_hist'] = macd_hist
                    # Add MACD crossover signal
                    df['macd_crossover'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
                
                elif feature == 'bbands':
                    # Bollinger Bands
                    periods = [5, 20]
                    for period in periods:
                        upper, middle, lower = talib.BBANDS(
                            close_prices, timeperiod=period, nbdevup=2, nbdevdn=2, matype=0
                        )
                        df[f'bbands_upper_{period}'] = upper
                        df[f'bbands_middle_{period}'] = middle
                        df[f'bbands_lower_{period}'] = lower
                        # Add Bollinger Band width
                        df[f'bbands_width_{period}'] = (upper - lower) / middle
                        # Add Bollinger Band position
                        df[f'bbands_pos_{period}'] = (df['close'] - lower) / (upper - lower)
                
                elif feature == 'atr':
                    # Average True Range
                    periods = [7, 14, 21]
                    for period in periods:
                        df[f'atr_{period}'] = talib.ATR(
                            high_prices, low_prices, close_prices, timeperiod=period
                        )
                        # Normalize ATR
                        df[f'atr_pct_{period}'] = df[f'atr_{period}'] / df['close']
                
                elif feature == 'stochastic':
                    # Stochastic Oscillator
                    k_periods = [14]
                    for k_period in k_periods:
                        slowk, slowd = talib.STOCH(
                            high_prices, low_prices, close_prices, 
                            fastk_period=k_period, slowk_period=3, slowk_matype=0, 
                            slowd_period=3, slowd_matype=0
                        )
                        df[f'stoch_k_{k_period}'] = slowk
                        df[f'stoch_d_{k_period}'] = slowd
                        # Add crossover
                        df[f'stoch_crossover_{k_period}'] = np.where(
                            df[f'stoch_k_{k_period}'] > df[f'stoch_d_{k_period}'], 1, -1
                        )
                
                elif feature == 'price_channel':
                    # Price Channel
                    periods = [20, 50]
                    for period in periods:
                        df[f'highest_high_{period}'] = df['high'].rolling(period).max()
                        df[f'lowest_low_{period}'] = df['low'].rolling(period).min()
                        # Add channel width
                        df[f'channel_width_{period}'] = (
                            df[f'highest_high_{period}'] - df[f'lowest_low_{period}']
                        ) / df['close']
                
                elif feature == 'momentum':
                    # Momentum
                    periods = [10, 20]
                    for period in periods:
                        df[f'momentum_{period}'] = df['close'].pct_change(period)
                
            except Exception as e:
                logger.error(f"Error creating price feature '{feature}': {str(e)}")
        
        return df
    
    def _create_volume_features(self, df):
        """
        Create volume-based technical indicators.
        
        Args:
            df (pandas.DataFrame): DataFrame containing OHLCV data
            
        Returns:
            pandas.DataFrame: DataFrame with added volume-based features
        """
        logger.info("Creating volume-based features")
        
        # Extract columns for easier use
        high_prices = df['high'].values
        low_prices = df['low'].values
        close_prices = df['close'].values
        volume = df['volume'].values
        
        for feature in self.volume_features:
            try:
                if feature == 'vwap':
                    # Volume Weighted Average Price
                    typical_price = (df['high'] + df['low'] + df['close']) / 3
                    daily_pv = typical_price * df['volume']
                    
                    # Reset at the start of each trading day
                    if isinstance(df.index, pd.DatetimeIndex):
                        # Group by date and calculate cumulative sums within each day
                        df['date_only'] = df.index.date
                        cum_pv = df.groupby('date_only')['daily_pv'].cumsum()
                        cum_vol = df.groupby('date_only')['volume'].cumsum()
                        df['vwap'] = cum_pv / cum_vol
                        df.drop('date_only', axis=1, inplace=True)
                    else:
                        # Simple cumulative calculation if no date index
                        df['vwap'] = daily_pv.cumsum() / df['volume'].cumsum()
                
                elif feature == 'obv':
                    # On-Balance Volume
                    df['obv'] = talib.OBV(close_prices, volume)
                    # Add OBV rate of change
                    df['obv_roc'] = df['obv'].pct_change(10)
                
                elif feature == 'ad':
                    # Accumulation/Distribution Line
                    df['ad'] = talib.AD(high_prices, low_prices, close_prices, volume)
                    # Add A/D rate of change
                    df['ad_roc'] = df['ad'].pct_change(10)
                
                elif feature == 'volume_ma':
                    # Volume Moving Averages
                    periods = [5, 10, 20]
                    for period in periods:
                        df[f'volume_sma_{period}'] = talib.SMA(volume, timeperiod=period)
                        # Add volume ratio
                        df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_sma_{period}']
                
                elif feature == 'volume_oscillator':
                    # Volume Oscillator (difference between fast and slow volume MAs)
                    df['volume_sma_5'] = talib.SMA(volume, timeperiod=5)
                    df['volume_sma_10'] = talib.SMA(volume, timeperiod=10)
                    df['volume_oscillator'] = (
                        (df['volume_sma_5'] - df['volume_sma_10']) / df['volume_sma_10'] * 100
                    )
                
                elif feature == 'mfi':
                    # Money Flow Index
                    periods = [14]
                    for period in periods:
                        df[f'mfi_{period}'] = talib.MFI(
                            high_prices, low_prices, close_prices, volume, timeperiod=period
                        )
                
                elif feature == 'chaikin_ad':
                    # Chaikin A/D Oscillator
                    df['chaikin_ad'] = talib.ADOSC(
                        high_prices, low_prices, close_prices, volume, fastperiod=3, slowperiod=10
                    )
                
            except Exception as e:
                logger.error(f"Error creating volume feature '{feature}': {str(e)}")
        
        return df
    
    def _create_return_features(self, df):
        """
        Create return-based features.
        
        Args:
            df (pandas.DataFrame): DataFrame containing OHLCV data
            
        Returns:
            pandas.DataFrame: DataFrame with added return-based features
        """
        logger.info("Creating return-based features")
        
        for feature in self.return_features:
            try:
                if feature == 'daily_return':
                    # Daily return
                    df['daily_return'] = df['close'].pct_change()
                
                elif feature == 'log_return':
                    # Log return
                    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
                
                elif feature == 'volatility':
                    # Volatility (rolling standard deviation of returns)
                    periods = [5, 10, 20]
                    for period in periods:
                        df[f'volatility_{period}'] = df['daily_return'].rolling(period).std()
                
                elif feature == 'sharpe':
                    # Rolling Sharpe ratio
                    risk_free_rate = self.config.get('feature_engineering', {}).get('risk_free_rate', 0.0)
                    periods = [20, 60]
                    for period in periods:
                        excess_return = df['daily_return'] - risk_free_rate / 252  # Daily risk-free rate
                        df[f'sharpe_{period}'] = (
                            excess_return.rolling(period).mean() / 
                            df['daily_return'].rolling(period).std()
                        ) * np.sqrt(252)  # Annualized
                
                elif feature == 'drawdown':
                    # Rolling maximum drawdown
                    periods = [20, 60]
                    for period in periods:
                        roll_max = df['close'].rolling(period).max()
                        drawdown = (df['close'] / roll_max - 1.0)
                        df[f'drawdown_{period}'] = drawdown
                
                elif feature == 'returns_skew':
                    # Return skewness
                    periods = [20, 60]
                    for period in periods:
                        df[f'returns_skew_{period}'] = df['daily_return'].rolling(period).apply(
                            lambda x: stats.skew(x)
                        )
                
                elif feature == 'returns_kurt':
                    # Return kurtosis
                    periods = [20, 60]
                    for period in periods:
                        df[f'returns_kurt_{period}'] = df['daily_return'].rolling(period).apply(
                            lambda x: stats.kurtosis(x)
                        )
                
                elif feature == 'updown_volatility':
                    # Upside and downside volatility
                    periods = [20]
                    for period in periods:
                        up_returns = df['daily_return'].copy()
                        up_returns[up_returns < 0] = 0
                        down_returns = df['daily_return'].copy()
                        down_returns[down_returns > 0] = 0
                        
                        df[f'upside_vol_{period}'] = up_returns.rolling(period).std()
                        df[f'downside_vol_{period}'] = down_returns.rolling(period).std().abs()
                        df[f'volatility_ratio_{period}'] = (
                            df[f'upside_vol_{period}'] / df[f'downside_vol_{period}']
                        )
                
            except Exception as e:
                logger.error(f"Error creating return feature '{feature}': {str(e)}")
        
        return df
    
    def _create_lagged_features(self, df):
        """
        Create lagged features for time series analysis.
        
        Args:
            df (pandas.DataFrame): DataFrame containing features
            
        Returns:
            pandas.DataFrame: DataFrame with added lagged features
        """
        logger.info("Creating lagged features")
        
        # Copy the dataframe to avoid modifying the original
        result_df = df.copy()
        
        # Columns to exclude from lagging
        exclude_columns = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        columns_to_lag = [col for col in df.columns if col not in exclude_columns]
        
        # Create lagged features for each period
        for period in self.lag_periods:
            try:
                for col in columns_to_lag:
                    result_df[f'{col}_lag_{period}'] = df[col].shift(period)
            except Exception as e:
                logger.error(f"Error creating lag {period} for features: {str(e)}")
        
        return result_df
    
    def normalize_features(self, df, method='zscore', exclude_cols=None):
        """
        Normalize features using different methods.
        
        Args:
            df (pandas.DataFrame): DataFrame with features
            method (str): Normalization method ('zscore', 'minmax', 'robust')
            exclude_cols (list): Columns to exclude from normalization
            
        Returns:
            pandas.DataFrame: DataFrame with normalized features
        """
        logger.info(f"Normalizing features using {method} method")
        
        if df is None or df.empty:
            return df
        
        # Copy dataframe to avoid modifying the original
        result_df = df.copy()
        
        # Default columns to exclude
        if exclude_cols is None:
            exclude_cols = ['date', 'symbol']
        
        # Get columns to normalize
        cols_to_normalize = [col for col in df.columns if col not in exclude_cols]
        
        # Perform normalization
        try:
            if method == 'zscore':
                # Z-score normalization (mean=0, std=1)
                for col in cols_to_normalize:
                    mean = df[col].mean()
                    std = df[col].std()
                    if std > 0:  # Avoid division by zero
                        result_df[col] = (df[col] - mean) / std
            
            elif method == 'minmax':
                # Min-max normalization (0 to 1 range)
                for col in cols_to_normalize:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    if max_val > min_val:  # Avoid division by zero
                        result_df[col] = (df[col] - min_val) / (max_val - min_val)
            
            elif method == 'robust':
                # Robust normalization using quantiles
                for col in cols_to_normalize:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    if iqr > 0:  # Avoid division by zero
                        result_df[col] = (df[col] - q1) / iqr
            
            else:
                logger.warning(f"Unknown normalization method: {method}")
                return df
        
        except Exception as e:
            logger.error(f"Error normalizing features: {str(e)}")
            return df
        
        return result_df
    
    def select_features(self, df, method='correlation', target_col='daily_return', top_n=20):
        """
        Select top features based on different methods.
        
        Args:
            df (pandas.DataFrame): DataFrame with features
            method (str): Feature selection method ('correlation', 'importance', 'variance')
            target_col (str): Target column for correlation-based selection
            top_n (int): Number of top features to select
            
        Returns:
            list: List of selected feature names
        """
        logger.info(f"Selecting top {top_n} features using {method} method")
        
        if df is None or df.empty:
            return []
        
        # Default columns to exclude from selection
        exclude_cols = ['date', 'symbol', target_col]
        
        # Get columns to consider for selection
        cols_to_select = [col for col in df.columns if col not in exclude_cols]
        
        try:
            if method == 'correlation':
                # Select features based on correlation with target
                if target_col not in df.columns:
                    logger.error(f"Target column '{target_col}' not found in DataFrame")
                    return []
                
                # Calculate correlation with target
                correlations = df[cols_to_select].corrwith(df[target_col]).abs()
                
                # Select top N features
                top_features = correlations.nlargest(top_n).index.tolist()
                
                return top_features
            
            elif method == 'variance':
                # Select features based on variance
                variances = df[cols_to_select].var()
                
                # Select top N features
                top_features = variances.nlargest(top_n).index.tolist()
                
                return top_features
            
            elif method == 'importance':
                # Feature importance from a tree-based model
                # Import here to avoid dependency issues
                from sklearn.ensemble import RandomForestRegressor
                
                if target_col not in df.columns:
                    logger.error(f"Target column '{target_col}' not found in DataFrame")
                    return []
                
                # Prepare data
                X = df[cols_to_select].dropna()
                y = df.loc[X.index, target_col]
                
                # Train a model
                rf = RandomForestRegressor(n_estimators=50, random_state=42)
                rf.fit(X, y)
                
                # Get feature importance
                importance = pd.Series(rf.feature_importances_, index=X.columns)
                
                # Select top N features
                top_features = importance.nlargest(top_n).index.tolist()
                
                return top_features
            
            else:
                logger.warning(f"Unknown feature selection method: {method}")
                return []
        
        except Exception as e:
            logger.error(f"Error selecting features: {str(e)}")
            return []
    
    def create_target_variables(self, df, horizon=5, method='return'):
        """
        Create target variables for supervised learning.
        
        Args:
            df (pandas.DataFrame): DataFrame with OHLCV data
            horizon (int): Prediction horizon in days
            method (str): Target variable method ('return', 'direction', 'volatility')
            
        Returns:
            pandas.DataFrame: DataFrame with added target variables
        """
        logger.info(f"Creating target variable with {horizon}-day horizon using {method} method")
        
        if df is None or df.empty:
            return df
        
        # Copy dataframe to avoid modifying the original
        result_df = df.copy()
        
        try:
            if method == 'return':
                # Future return
                result_df[f'target_{horizon}d_return'] = (
                    result_df['close'].shift(-horizon) / result_df['close'] - 1
                )
            
            elif method == 'direction':
                # Future price direction (binary)
                future_returns = result_df['close'].shift(-horizon) / result_df['close'] - 1
                result_df[f'target_{horizon}d_direction'] = (future_returns > 0).astype(int)
            
            elif method == 'volatility':
                # Future volatility
                future_returns = result_df['close'].shift(-horizon) / result_df['close'] - 1
                result_df[f'target_{horizon}d_volatility'] = (
                    future_returns.rolling(window=horizon).std().shift(-horizon)
                )
            
            else:
                logger.warning(f"Unknown target variable method: {method}")
        
        except Exception as e:
            logger.error(f"Error creating target variable: {str(e)}")
        
        return result_df
    
    def save_features(self, df, symbol, prefix='features'):
        """
        Save engineered features to database.
        
        Args:
            df (pandas.DataFrame): DataFrame with features
            symbol (str): Stock symbol
            prefix (str): Table name prefix
            
        Returns:
            bool: True if successful, False otherwise
        """
        if df is None or df.empty:
            logger.warning(f"No feature data to save for {symbol}")
            return False
        
        table_name = f"{prefix}_{symbol.lower()}"
        
        try:
            self.db_connector.save_dataframe(df, table_name)
            logger.info(f"Saved {len(df)} feature records for {symbol} to database")
            return True
        except Exception as e:
            logger.error(f"Error saving features for {symbol} to database: {str(e)}")
            return False
    
    def load_features(self, symbol, start_date=None, end_date=None, prefix='features'):
        """
        Load engineered features from database.
        
        Args:
            symbol (str): Stock symbol
            start_date (str, optional): Start date in format 'YYYY-MM-DD'
            end_date (str, optional): End date in format 'YYYY-MM-DD'
            prefix (str): Table name prefix
            
        Returns:
            pandas.DataFrame: DataFrame containing features
        """
        table_name = f"{prefix}_{symbol.lower()}"
        
        query = f"SELECT * FROM {table_name}"
        conditions = []
        
        if start_date:
            conditions.append(f"date >= '{start_date}'")
        if end_date:
            conditions.append(f"date <= '{end_date}'")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        try:
            df = self.db_connector.run_query(query)
            
            # Convert date to datetime and set as index
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Error loading features for {symbol} from database: {str(e)}")
            return None 