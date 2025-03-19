"""
Economic Data Fetcher

This module handles fetching economic indicator data for analysis.
"""

import os
import time
import pandas as pd
import requests
from datetime import datetime, timedelta
import logging
from ..utils.database import DatabaseConnector

logger = logging.getLogger('marketmind.data_acquisition.economic_data')

class EconomicDataFetcher:
    """
    Fetches economic indicator data from Alpha Vantage and other sources.
    """
    
    def __init__(self, config=None):
        """
        Initialize the EconomicDataFetcher with configuration.
        
        Args:
            config (dict): Configuration dictionary containing API keys and settings.
        """
        from .. import load_config
        self.config = config or load_config()
        self.api_key = self.config.get('api', {}).get('alpha_vantage', {}).get('key')
        self.base_url = self.config.get('api', {}).get('alpha_vantage', {}).get('base_url', 'https://www.alphavantage.co/query')
        self.call_limit = self.config.get('api', {}).get('alpha_vantage', {}).get('call_limit_per_minute', 5)
        self.indicators = self.config.get('data', {}).get('economic_indicators', {}).get('include', [])
        self.db_connector = DatabaseConnector(self.config)
        
        if not self.api_key or self.api_key == "YOUR_ALPHA_VANTAGE_API_KEY":
            logger.warning("Alpha Vantage API key not set. Please set your API key in config.yaml")
    
    def fetch_economic_indicator(self, indicator, save=True):
        """
        Fetch an economic indicator from Alpha Vantage.
        
        Args:
            indicator (str): Economic indicator to fetch (e.g., 'FEDFUNDS', 'UNRATE')
            save (bool): Whether to save the data to database
            
        Returns:
            pandas.DataFrame: DataFrame containing the indicator data
        """
        if not self.api_key:
            logger.error("Cannot fetch data: API key not set")
            return None
        
        logger.info(f"Fetching economic indicator: {indicator}")
        
        params = {
            'function': 'ECONOMIC_INDICATOR',
            'name': indicator,
            'apikey': self.api_key,
            'datatype': 'json'
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Check for error messages
            if 'Error Message' in data:
                logger.error(f"API returned error: {data['Error Message']}")
                return None
            
            # Extract data
            data_points = data.get('data', [])
            
            if not data_points:
                logger.warning(f"No data returned for indicator: {indicator}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data_points)
            
            # Rename columns
            df.columns = ['date', 'value']
            
            # Convert values to numeric
            df['value'] = pd.to_numeric(df['value'])
            
            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Set date as index
            df.set_index('date', inplace=True)
            
            # Add indicator column
            df['indicator'] = indicator
            
            # Sort by date (ascending)
            df.sort_index(inplace=True)
            
            # Save to database if requested
            if save:
                self._save_to_database(df, indicator)
            
            logger.info(f"Successfully fetched {len(df)} data points for indicator: {indicator}")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching indicator data for {indicator}: {str(e)}")
            return None
        
        except ValueError as e:
            logger.error(f"Error parsing JSON response for {indicator}: {str(e)}")
            return None
    
    def fetch_federal_funds_rate(self, save=True):
        """
        Fetch Federal Funds Rate data.
        
        Args:
            save (bool): Whether to save the data to database
            
        Returns:
            pandas.DataFrame: DataFrame containing the Federal Funds Rate data
        """
        return self.fetch_economic_indicator('FEDFUNDS', save)
    
    def fetch_unemployment_rate(self, save=True):
        """
        Fetch Unemployment Rate data.
        
        Args:
            save (bool): Whether to save the data to database
            
        Returns:
            pandas.DataFrame: DataFrame containing the Unemployment Rate data
        """
        return self.fetch_economic_indicator('UNRATE', save)
    
    def fetch_cpi(self, save=True):
        """
        Fetch Consumer Price Index (CPI) data.
        
        Args:
            save (bool): Whether to save the data to database
            
        Returns:
            pandas.DataFrame: DataFrame containing the CPI data
        """
        return self.fetch_economic_indicator('CPI', save)
    
    def fetch_gdp(self, save=True):
        """
        Fetch GDP data.
        
        Args:
            save (bool): Whether to save the data to database
            
        Returns:
            pandas.DataFrame: DataFrame containing the GDP data
        """
        return self.fetch_economic_indicator('GDP', save)
    
    def fetch_retail_sales(self, save=True):
        """
        Fetch Retail Sales data.
        
        Args:
            save (bool): Whether to save the data to database
            
        Returns:
            pandas.DataFrame: DataFrame containing the Retail Sales data
        """
        return self.fetch_economic_indicator('RETAILSALES', save)
    
    def fetch_all_indicators(self, save=True):
        """
        Fetch all configured economic indicators.
        
        Args:
            save (bool): Whether to save the data to database
            
        Returns:
            dict: Dictionary of DataFrames keyed by indicator
        """
        results = {}
        
        for i, indicator in enumerate(self.indicators):
            # Respect API rate limits
            if i > 0 and i % self.call_limit == 0:
                logger.info(f"Reached API call limit. Pausing for 60 seconds...")
                time.sleep(60)
            
            df = self.fetch_economic_indicator(indicator, save)
            if df is not None:
                results[indicator] = df
        
        return results
    
    def _save_to_database(self, df, indicator):
        """
        Save the fetched data to database.
        
        Args:
            df (pandas.DataFrame): DataFrame to save
            indicator (str): Indicator name
        """
        try:
            table_name = f"economic_{indicator.lower()}"
            self.db_connector.save_dataframe(df, table_name)
            logger.info(f"Saved {indicator} data to database")
        except Exception as e:
            logger.error(f"Error saving {indicator} data to database: {str(e)}")
    
    def load_from_database(self, indicator, start_date=None, end_date=None):
        """
        Load economic indicator data from database.
        
        Args:
            indicator (str): Indicator name
            start_date (str, optional): Start date in format 'YYYY-MM-DD'
            end_date (str, optional): End date in format 'YYYY-MM-DD'
            
        Returns:
            pandas.DataFrame: DataFrame containing the indicator data
        """
        table_name = f"economic_{indicator.lower()}"
        
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
            return df
        except Exception as e:
            logger.error(f"Error loading {indicator} data from database: {str(e)}")
            return None
    
    def merge_with_stock_data(self, stock_df, indicators=None, interpolate=True):
        """
        Merge economic indicators with stock data.
        
        Args:
            stock_df (pandas.DataFrame): Stock data DataFrame with datetime index
            indicators (list, optional): List of indicators to include, defaults to all configured
            interpolate (bool): Whether to interpolate missing values
            
        Returns:
            pandas.DataFrame: DataFrame with stock data and economic indicators
        """
        if not isinstance(stock_df.index, pd.DatetimeIndex):
            logger.error("Stock data must have a DatetimeIndex")
            return stock_df
        
        if indicators is None:
            indicators = self.indicators
        
        result_df = stock_df.copy()
        
        for indicator in indicators:
            try:
                # Load indicator data
                indicator_df = self.load_from_database(indicator)
                
                if indicator_df is None or indicator_df.empty:
                    logger.warning(f"No data available for indicator: {indicator}")
                    continue
                
                # Ensure the indicator data has a datetime index
                if not isinstance(indicator_df.index, pd.DatetimeIndex):
                    if 'date' in indicator_df.columns:
                        indicator_df['date'] = pd.to_datetime(indicator_df['date'])
                        indicator_df.set_index('date', inplace=True)
                    else:
                        logger.warning(f"Cannot set datetime index for indicator: {indicator}")
                        continue
                
                # Rename value column to avoid conflicts
                if 'value' in indicator_df.columns:
                    indicator_df.rename(columns={'value': f'{indicator.lower()}_value'}, inplace=True)
                
                # Filter columns to include
                indicator_cols = [col for col in indicator_df.columns if col not in ['indicator']]
                indicator_df = indicator_df[indicator_cols]
                
                # Merge with stock data on date index
                result_df = pd.merge(result_df, indicator_df, 
                                      left_index=True, right_index=True, 
                                      how='left')
                
                # Interpolate missing values if requested
                if interpolate:
                    for col in indicator_cols:
                        result_df[col] = result_df[col].interpolate(method='time')
                
                logger.info(f"Merged indicator: {indicator}")
                
            except Exception as e:
                logger.error(f"Error merging indicator {indicator}: {str(e)}")
        
        return result_df 