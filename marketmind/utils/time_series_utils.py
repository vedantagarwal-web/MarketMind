"""
Time Series Utilities

This module provides utilities for time series data analysis and processing.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import List, Tuple, Union, Optional

logger = logging.getLogger('marketmind.utils.time_series_utils')

class TimeSeriesHelper:
    """
    Helper utilities for time series data.
    """
    
    def __init__(self, config=None):
        """
        Initialize the time series helper.
        
        Args:
            config (dict): Configuration dictionary.
        """
        from .. import load_config
        self.config = config or load_config()
    
    def create_sequences(self, data: np.ndarray, sequence_length: int, target_idx: int = -1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.
        
        Args:
            data (numpy.ndarray): Input data
            sequence_length (int): Length of each sequence
            target_idx (int): Index of target feature in data
            
        Returns:
            tuple: (X, y) where X is sequences and y is targets
        """
        X, y = [], []
        
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length, target_idx])
        
        return np.array(X), np.array(y)
    
    def create_multistep_sequences(self, data: np.ndarray, sequence_length: int, 
                                  forecast_horizon: int, target_idx: int = -1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for multi-step time series prediction.
        
        Args:
            data (numpy.ndarray): Input data
            sequence_length (int): Length of each sequence
            forecast_horizon (int): Number of steps to forecast
            target_idx (int): Index of target feature in data
            
        Returns:
            tuple: (X, y) where X is sequences and y is targets
        """
        X, y = [], []
        
        for i in range(len(data) - sequence_length - forecast_horizon + 1):
            X.append(data[i:(i + sequence_length)])
            y.append(data[(i + sequence_length):(i + sequence_length + forecast_horizon), target_idx])
        
        return np.array(X), np.array(y)
    
    def apply_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features to a DataFrame with datetime index.
        
        Args:
            df (pandas.DataFrame): DataFrame with datetime index
            
        Returns:
            pandas.DataFrame: DataFrame with added time features
        """
        # Check for datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("DataFrame does not have a DatetimeIndex. Cannot add time features.")
            return df
        
        df = df.copy()
        
        # Add day of week (0-6, 0 is Monday)
        df['day_of_week'] = df.index.dayofweek
        
        # Add day of month (1-31)
        df['day_of_month'] = df.index.day
        
        # Add day of year (1-366)
        df['day_of_year'] = df.index.dayofyear
        
        # Add week of year (1-53)
        df['week_of_year'] = df.index.isocalendar().week
        
        # Add month (1-12)
        df['month'] = df.index.month
        
        # Add quarter (1-4)
        df['quarter'] = df.index.quarter
        
        # Add year
        df['year'] = df.index.year
        
        # Add is_month_start, is_month_end, is_quarter_start, is_quarter_end, is_year_start, is_year_end
        df['is_month_start'] = df.index.is_month_start.astype(int)
        df['is_month_end'] = df.index.is_month_end.astype(int)
        df['is_quarter_start'] = df.index.is_quarter_start.astype(int)
        df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        df['is_year_start'] = df.index.is_year_start.astype(int)
        df['is_year_end'] = df.index.is_year_end.astype(int)
        
        # Add is_weekend (0 or 1)
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        # Add sinusoidal features for cyclical variables (day of week, month, etc.)
        # This helps capture the cyclical nature of time
        df['day_of_week_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
        
        return df
    
    def align_to_business_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Align time series data to business days (Mon-Fri), handling missing days.
        
        Args:
            df (pandas.DataFrame): DataFrame with datetime index
            
        Returns:
            pandas.DataFrame: DataFrame with complete business day index
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("DataFrame does not have a DatetimeIndex. Cannot align to business days.")
            return df
        
        # Get date range for business days
        start_date = df.index.min()
        end_date = df.index.max()
        
        # Create full business day date range
        business_days = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # Reindex data to fill in missing business days
        df_aligned = df.reindex(business_days)
        
        # Log info about missing days
        missing_days = len(business_days) - df.index.isin(business_days).sum()
        if missing_days > 0:
            logger.info(f"Filled in {missing_days} missing business days")
        
        return df_aligned
    
    def get_lagged_features(self, df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
        """
        Create lagged features for specified columns.
        
        Args:
            df (pandas.DataFrame): DataFrame with time series data
            columns (list): List of column names to create lags for
            lags (list): List of lag periods
            
        Returns:
            pandas.DataFrame: DataFrame with added lagged features
        """
        df_result = df.copy()
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found. Skipping lag features.")
                continue
                
            for lag in lags:
                lag_name = f"{col}_lag_{lag}"
                df_result[lag_name] = df[col].shift(lag)
        
        return df_result
    
    def get_rolling_features(self, df: pd.DataFrame, columns: List[str], 
                             windows: List[int], functions: List[str]) -> pd.DataFrame:
        """
        Create rolling window features for specified columns.
        
        Args:
            df (pandas.DataFrame): DataFrame with time series data
            columns (list): List of column names to create rolling features for
            windows (list): List of window sizes
            functions (list): List of functions to apply ('mean', 'std', 'min', 'max', etc.)
            
        Returns:
            pandas.DataFrame: DataFrame with added rolling features
        """
        df_result = df.copy()
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found. Skipping rolling features.")
                continue
                
            for window in windows:
                for function in functions:
                    feature_name = f"{col}_rolling_{window}_{function}"
                    
                    if function == 'mean':
                        df_result[feature_name] = df[col].rolling(window=window).mean()
                    elif function == 'std':
                        df_result[feature_name] = df[col].rolling(window=window).std()
                    elif function == 'min':
                        df_result[feature_name] = df[col].rolling(window=window).min()
                    elif function == 'max':
                        df_result[feature_name] = df[col].rolling(window=window).max()
                    elif function == 'median':
                        df_result[feature_name] = df[col].rolling(window=window).median()
                    elif function == 'var':
                        df_result[feature_name] = df[col].rolling(window=window).var()
                    elif function == 'skew':
                        df_result[feature_name] = df[col].rolling(window=window).skew()
                    elif function == 'kurt':
                        df_result[feature_name] = df[col].rolling(window=window).kurt()
                    else:
                        logger.warning(f"Unknown rolling function: {function}")
        
        return df_result
    
    def get_expanding_features(self, df: pd.DataFrame, columns: List[str], 
                               min_periods: int, functions: List[str]) -> pd.DataFrame:
        """
        Create expanding window features for specified columns.
        
        Args:
            df (pandas.DataFrame): DataFrame with time series data
            columns (list): List of column names to create expanding features for
            min_periods (int): Minimum number of observations required
            functions (list): List of functions to apply ('mean', 'std', 'min', 'max', etc.)
            
        Returns:
            pandas.DataFrame: DataFrame with added expanding features
        """
        df_result = df.copy()
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found. Skipping expanding features.")
                continue
                
            for function in functions:
                feature_name = f"{col}_expanding_{function}"
                
                if function == 'mean':
                    df_result[feature_name] = df[col].expanding(min_periods=min_periods).mean()
                elif function == 'std':
                    df_result[feature_name] = df[col].expanding(min_periods=min_periods).std()
                elif function == 'min':
                    df_result[feature_name] = df[col].expanding(min_periods=min_periods).min()
                elif function == 'max':
                    df_result[feature_name] = df[col].expanding(min_periods=min_periods).max()
                elif function == 'median':
                    df_result[feature_name] = df[col].expanding(min_periods=min_periods).median()
                elif function == 'var':
                    df_result[feature_name] = df[col].expanding(min_periods=min_periods).var()
                elif function == 'skew':
                    df_result[feature_name] = df[col].expanding(min_periods=min_periods).skew()
                elif function == 'kurt':
                    df_result[feature_name] = df[col].expanding(min_periods=min_periods).kurt()
                else:
                    logger.warning(f"Unknown expanding function: {function}")
        
        return df_result
    
    def get_return_features(self, df: pd.DataFrame, columns: List[str], 
                            periods: List[int]) -> pd.DataFrame:
        """
        Create return features for specified columns.
        
        Args:
            df (pandas.DataFrame): DataFrame with time series data
            columns (list): List of column names to create return features for
            periods (list): List of periods for returns
            
        Returns:
            pandas.DataFrame: DataFrame with added return features
        """
        df_result = df.copy()
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found. Skipping return features.")
                continue
                
            for period in periods:
                # Simple returns
                simple_return_name = f"{col}_return_{period}"
                df_result[simple_return_name] = df[col].pct_change(periods=period)
                
                # Log returns
                log_return_name = f"{col}_log_return_{period}"
                df_result[log_return_name] = np.log(df[col] / df[col].shift(period))
        
        return df_result
    
    def get_diff_features(self, df: pd.DataFrame, columns: List[str], 
                          periods: List[int]) -> pd.DataFrame:
        """
        Create difference features for specified columns.
        
        Args:
            df (pandas.DataFrame): DataFrame with time series data
            columns (list): List of column names to create difference features for
            periods (list): List of periods for differences
            
        Returns:
            pandas.DataFrame: DataFrame with added difference features
        """
        df_result = df.copy()
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found. Skipping difference features.")
                continue
                
            for period in periods:
                diff_name = f"{col}_diff_{period}"
                df_result[diff_name] = df[col].diff(periods=period)
                
                pct_diff_name = f"{col}_pct_diff_{period}"
                df_result[pct_diff_name] = df[col].pct_change(periods=period)
        
        return df_result
    
    def get_ewm_features(self, df: pd.DataFrame, columns: List[str], 
                         spans: List[int], functions: List[str]) -> pd.DataFrame:
        """
        Create exponentially weighted features for specified columns.
        
        Args:
            df (pandas.DataFrame): DataFrame with time series data
            columns (list): List of column names to create EWM features for
            spans (list): List of span values for EWM
            functions (list): List of functions to apply ('mean', 'std', 'var')
            
        Returns:
            pandas.DataFrame: DataFrame with added EWM features
        """
        df_result = df.copy()
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found. Skipping EWM features.")
                continue
                
            for span in spans:
                for function in functions:
                    feature_name = f"{col}_ewm_{span}_{function}"
                    
                    if function == 'mean':
                        df_result[feature_name] = df[col].ewm(span=span, adjust=False).mean()
                    elif function == 'std':
                        df_result[feature_name] = df[col].ewm(span=span, adjust=False).std()
                    elif function == 'var':
                        df_result[feature_name] = df[col].ewm(span=span, adjust=False).var()
                    else:
                        logger.warning(f"Unknown EWM function: {function}")
        
        return df_result
    
    def get_seasonal_decomposition(self, df: pd.DataFrame, column: str, 
                                   period: int, model: str = 'additive') -> pd.DataFrame:
        """
        Perform seasonal decomposition on time series data.
        
        Args:
            df (pandas.DataFrame): DataFrame with time series data
            column (str): Column name to decompose
            period (int): Period for seasonal decomposition
            model (str): Type of seasonal decomposition ('additive' or 'multiplicative')
            
        Returns:
            pandas.DataFrame: DataFrame with decomposition components
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("DataFrame does not have a DatetimeIndex. Cannot perform seasonal decomposition.")
            return df
        
        if column not in df.columns:
            logger.warning(f"Column '{column}' not found. Cannot perform seasonal decomposition.")
            return df
        
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Fill any missing values (required for decomposition)
        filled_data = df[column].fillna(method='ffill').fillna(method='bfill')
        
        try:
            # Perform decomposition
            result = seasonal_decompose(filled_data, model=model, period=period)
            
            # Create result DataFrame
            result_df = pd.DataFrame({
                f"{column}_trend": result.trend,
                f"{column}_seasonal": result.seasonal,
                f"{column}_residual": result.resid
            })
            
            # Merge with original DataFrame
            df_result = pd.concat([df, result_df], axis=1)
            
            return df_result
            
        except Exception as e:
            logger.error(f"Error performing seasonal decomposition: {str(e)}")
            return df
    
    def detect_anomalies(self, series: pd.Series, window: int = 20, 
                         n_sigmas: float = 3.0) -> pd.Series:
        """
        Detect anomalies in time series data using a rolling mean and standard deviation.
        
        Args:
            series (pandas.Series): Time series data
            window (int): Window size for rolling statistics
            n_sigmas (float): Number of standard deviations for anomaly threshold
            
        Returns:
            pandas.Series: Boolean series with True for anomalies
        """
        # Calculate rolling mean and standard deviation
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        
        # Calculate lower and upper bounds
        lower_bound = rolling_mean - (n_sigmas * rolling_std)
        upper_bound = rolling_mean + (n_sigmas * rolling_std)
        
        # Detect anomalies
        anomalies = (series < lower_bound) | (series > upper_bound)
        
        return anomalies 