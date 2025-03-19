"""
Data Validation

This module provides utilities for validating data.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger('marketmind.utils.data_validation')

class DataValidator:
    """
    Handles validation of data to ensure quality and consistency.
    """
    
    def __init__(self, config=None):
        """
        Initialize the data validator.
        
        Args:
            config (dict): Configuration dictionary.
        """
        from .. import load_config
        self.config = config or load_config()
    
    def validate_stock_data(self, df):
        """
        Validate stock data for completeness and correctness.
        
        Args:
            df (pandas.DataFrame): Stock data DataFrame
            
        Returns:
            tuple: (bool, str) - (is_valid, error_message)
        """
        if df is None or df.empty:
            return False, "DataFrame is empty or None"
        
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        alt_required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Check if required columns exist (with either capitalization)
        missing_columns = []
        for req_col, alt_col in zip(required_columns, alt_required_columns):
            if req_col not in df.columns and alt_col not in df.columns:
                missing_columns.append(f"{req_col}/{alt_col}")
        
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}"
        
        # Check for NaN values
        nan_counts = df.isna().sum()
        if nan_counts.sum() > 0:
            nan_info = ', '.join([f"{col}: {count}" for col, count in nan_counts.items() if count > 0])
            logger.warning(f"NaN values found: {nan_info}")
        
        # Check for duplicated dates
        if isinstance(df.index, pd.DatetimeIndex):
            dup_dates = df.index.duplicated()
            if dup_dates.any():
                dup_count = dup_dates.sum()
                logger.warning(f"Found {dup_count} duplicated dates in index")
        
        # Check for negative values in places where they shouldn't exist
        price_columns = [col for col in df.columns if col.lower() in ['open', 'high', 'low', 'close']]
        volume_columns = [col for col in df.columns if col.lower() == 'volume']
        
        for col in price_columns:
            if (df[col] <= 0).any():
                neg_count = (df[col] <= 0).sum()
                logger.warning(f"Found {neg_count} non-positive values in {col}")
        
        for col in volume_columns:
            if (df[col] < 0).any():
                neg_count = (df[col] < 0).sum()
                logger.warning(f"Found {neg_count} negative values in {col}")
        
        return True, "Data validation passed"
    
    def validate_news_data(self, df):
        """
        Validate news data for completeness and correctness.
        
        Args:
            df (pandas.DataFrame): News data DataFrame
            
        Returns:
            tuple: (bool, str) - (is_valid, error_message)
        """
        if df is None or df.empty:
            return False, "DataFrame is empty or None"
        
        # Check required columns
        required_columns = ['title', 'publishedAt', 'url']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}"
        
        # Check for NaN values in critical columns
        critical_columns = ['title', 'publishedAt']
        nan_counts = df[critical_columns].isna().sum()
        
        if nan_counts.sum() > 0:
            nan_info = ', '.join([f"{col}: {count}" for col, count in nan_counts.items() if count > 0])
            logger.warning(f"NaN values found in critical columns: {nan_info}")
            return False, f"NaN values found in critical columns: {nan_info}"
        
        return True, "Data validation passed"
    
    def clean_stock_data(self, df, inplace=False):
        """
        Clean stock data by handling missing values, duplicates, etc.
        
        Args:
            df (pandas.DataFrame): Stock data DataFrame
            inplace (bool): Whether to modify df in place or return a copy
            
        Returns:
            pandas.DataFrame: Cleaned DataFrame
        """
        if not inplace:
            df = df.copy()
        
        # Standardize column names
        column_map = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        
        for old_name, new_name in column_map.items():
            if old_name in df.columns and new_name not in df.columns:
                df[new_name] = df[old_name]
        
        # Handle NaN values
        # Forward fill missing values (use previous day's data)
        df.fillna(method='ffill', inplace=True)
        
        # If there are still NaN values at the beginning, use backward fill
        df.fillna(method='bfill', inplace=True)
        
        # For any remaining NaNs, use column median
        df.fillna(df.median(), inplace=True)
        
        # Remove duplicate indices if any
        if isinstance(df.index, pd.DatetimeIndex) and df.index.duplicated().any():
            df = df[~df.index.duplicated(keep='first')]
        
        # Ensure volume is at least 0
        if 'volume' in df.columns:
            df['volume'] = df['volume'].clip(lower=0)
        
        # Ensure price values are positive
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns:
                df[col] = df[col].clip(lower=0.01)
        
        # Sort by date
        if isinstance(df.index, pd.DatetimeIndex):
            df.sort_index(inplace=True)
        
        return df
    
    def detect_outliers(self, df, columns=None, method='zscore', threshold=3.0):
        """
        Detect outliers in the data.
        
        Args:
            df (pandas.DataFrame): DataFrame to check
            columns (list, optional): Columns to check for outliers
            method (str): Method to use ('zscore', 'iqr')
            threshold (float): Threshold for outlier detection
            
        Returns:
            pandas.DataFrame: DataFrame with boolean mask of outliers
        """
        if columns is None:
            # Use all numeric columns
            columns = df.select_dtypes(include=np.number).columns.tolist()
        
        outliers = pd.DataFrame(index=df.index)
        
        for col in columns:
            if col not in df.columns:
                continue
                
            if method == 'zscore':
                # Z-score method
                mean = df[col].mean()
                std = df[col].std()
                if std == 0:  # Handle zero std
                    outliers[col] = False
                    continue
                z_scores = (df[col] - mean) / std
                outliers[col] = abs(z_scores) > threshold
                
            elif method == 'iqr':
                # IQR method
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (threshold * iqr)
                upper_bound = q3 + (threshold * iqr)
                outliers[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
                
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")
        
        return outliers
    
    def handle_outliers(self, df, outliers, method='clip', inplace=False):
        """
        Handle outliers in the data.
        
        Args:
            df (pandas.DataFrame): DataFrame to handle outliers in
            outliers (pandas.DataFrame): DataFrame with boolean mask of outliers
            method (str): Method to use ('clip', 'mean', 'median', 'remove')
            inplace (bool): Whether to modify df in place
            
        Returns:
            pandas.DataFrame: DataFrame with handled outliers
        """
        if not inplace:
            df = df.copy()
        
        for col in outliers.columns:
            if col not in df.columns:
                continue
                
            # Get indices of outliers
            outlier_indices = outliers.index[outliers[col]]
            
            if len(outlier_indices) == 0:
                continue
                
            if method == 'clip':
                # Clip outliers to the threshold values
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (1.5 * iqr)
                upper_bound = q3 + (1.5 * iqr)
                df.loc[outlier_indices, col] = df.loc[outlier_indices, col].clip(lower=lower_bound, upper=upper_bound)
                
            elif method == 'mean':
                # Replace outliers with column mean
                mean_value = df[col].mean()
                df.loc[outlier_indices, col] = mean_value
                
            elif method == 'median':
                # Replace outliers with column median
                median_value = df[col].median()
                df.loc[outlier_indices, col] = median_value
                
            elif method == 'remove':
                # Remove rows with outliers (not recommended for time series)
                df = df.drop(index=outlier_indices)
                
            else:
                raise ValueError(f"Unknown outlier handling method: {method}")
        
        return df 