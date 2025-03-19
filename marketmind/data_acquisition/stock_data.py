"""
Stock Data Fetcher

This module handles fetching stock market data from various sources.
"""

import os
import time
import pandas as pd
import numpy as np
import logging
import requests
from datetime import datetime, timedelta

# Try importing yfinance, but provide graceful fallback if not installed
try:
    import yfinance as yf
    YAHOO_AVAILABLE = True
except ImportError:
    yf = None
    YAHOO_AVAILABLE = False
    logging.getLogger('marketmind.data_acquisition.stock_data').warning(
        "yfinance package not found. Yahoo Finance data source will not be available. "
        "Install with: pip install yfinance"
    )

from ..utils.database import DatabaseConnector

# Initialize logger
logger = logging.getLogger('marketmind.data_acquisition.stock_data')

class StockDataFetcher:
    """
    Fetches stock market data from various sources like Yahoo Finance, Alpha Vantage, etc.
    """
    
    def __init__(self, config=None):
        """
        Initialize the StockDataFetcher with configuration.
        
        Args:
            config (dict): Configuration dictionary containing API keys and settings.
        """
        from .. import load_config
        self.config = config or load_config()
        self.db_connector = DatabaseConnector(self.config)
        
        # Set up Alpha Vantage API configuration
        self.alpha_vantage_key = self.config.get('api', {}).get('alpha_vantage', {}).get('key')
        self.alpha_vantage_url = self.config.get('api', {}).get('alpha_vantage', {}).get('base_url', 'https://www.alphavantage.co/query')
        self.call_limit = self.config.get('api', {}).get('alpha_vantage', {}).get('call_limit_per_minute', 5)
        
        # Get default data provider, with fallback if preferred provider not available
        self.default_provider = self.config.get('data', {}).get('stocks', {}).get('default_provider', 'yahoo')
        if self.default_provider == 'yahoo' and not YAHOO_AVAILABLE:
            self.default_provider = 'alpha_vantage'
            logger.info("Yahoo Finance not available, falling back to Alpha Vantage")
        
        # Get watchlist
        self.watchlist = self.config.get('data', {}).get('stocks', {}).get('watchlist', [])
        
        if not self.watchlist:
            logger.warning("Stock watchlist is empty. Add stocks to your config file.")
    
    def fetch_stock_data(self, symbol, start_date=None, end_date=None, provider=None, interval='1d', save=True):
        """
        Fetch stock data for a specific symbol.
        
        Args:
            symbol (str): Stock symbol
            start_date (str, optional): Start date in format 'YYYY-MM-DD'
            end_date (str, optional): End date in format 'YYYY-MM-DD'
            provider (str, optional): Data provider ('yahoo', 'alpha_vantage')
            interval (str, optional): Data interval ('1d', '1h', etc.)
            save (bool): Whether to save the data to database
            
        Returns:
            pandas.DataFrame: DataFrame containing the stock data
        """
        # Set defaults if not provided
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if provider is None:
            provider = self.default_provider
        
        # Check if chosen provider is available
        if provider == 'yahoo' and not YAHOO_AVAILABLE:
            logger.warning("Yahoo Finance not available, falling back to Alpha Vantage")
            provider = 'alpha_vantage'
        
        logger.info(f"Fetching {interval} stock data for {symbol} from {start_date} to {end_date} using {provider}")
        
        # Try to get data from database first if save is True
        if save:
            db_data = self.load_from_database(symbol, start_date, end_date, interval)
            if db_data is not None and not db_data.empty:
                logger.info(f"Loaded {len(db_data)} data points for {symbol} from database")
                return db_data
        
        if provider == 'yahoo':
            return self._fetch_from_yahoo(symbol, start_date, end_date, interval, save)
        elif provider == 'alpha_vantage':
            return self._fetch_from_alpha_vantage(symbol, start_date, end_date, interval, save)
        else:
            logger.error(f"Unsupported data provider: {provider}")
            return None
    
    def _fetch_from_yahoo(self, symbol, start_date, end_date, interval='1d', save=True):
        """
        Fetch stock data from Yahoo Finance.
        
        Args:
            symbol (str): Stock symbol
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
            interval (str): Data interval ('1d', '1h', etc.)
            save (bool): Whether to save the data to database
            
        Returns:
            pandas.DataFrame: DataFrame containing the stock data
        """
        if not YAHOO_AVAILABLE:
            logger.error("Cannot fetch from Yahoo Finance: yfinance package not installed")
            return None
            
        try:
            # Convert dates to datetime
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            
            # Fetch data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, interval=interval)
            
            if df.empty:
                logger.warning(f"No data returned from Yahoo Finance for {symbol}")
                return None
            
            # Standardize column names
            df.columns = [col.lower() for col in df.columns]
            
            # Make sure 'volume' column exists
            if 'volume' not in df.columns:
                df['volume'] = 0
            
            # Add symbol column
            df['symbol'] = symbol
            
            # Save to database if requested
            if save:
                self._save_to_database(df, symbol, interval)
            
            logger.info(f"Successfully fetched {len(df)} data points for {symbol} from Yahoo Finance")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol} from Yahoo Finance: {str(e)}")
            return None
    
    def _fetch_from_alpha_vantage(self, symbol, start_date, end_date, interval='1d', save=True):
        """
        Fetch stock data from Alpha Vantage.
        
        Args:
            symbol (str): Stock symbol
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
            interval (str): Data interval ('1d', '1h', etc.)
            save (bool): Whether to save the data to database
            
        Returns:
            pandas.DataFrame: DataFrame containing the stock data
        """
        if not self.alpha_vantage_key:
            logger.error("Alpha Vantage API key not set")
            return None
        
        try:
            # Map interval to Alpha Vantage function
            interval_map = {
                '1d': 'TIME_SERIES_DAILY',
                '1h': 'TIME_SERIES_INTRADAY&interval=60min',
                '5min': 'TIME_SERIES_INTRADAY&interval=5min',
                '1min': 'TIME_SERIES_INTRADAY&interval=1min',
                'weekly': 'TIME_SERIES_WEEKLY',
                'monthly': 'TIME_SERIES_MONTHLY'
            }
            
            if interval not in interval_map:
                logger.error(f"Unsupported interval: {interval}")
                return None
            
            function = interval_map[interval]
            
            # Build API request parameters
            params = {
                'function': function.split('&')[0],
                'symbol': symbol,
                'apikey': self.alpha_vantage_key,
                'outputsize': 'full',
                'datatype': 'json'
            }
            
            # Add interval parameter if needed
            if '&' in function:
                params['interval'] = function.split('&')[1].split('=')[1]
            
            # Send API request
            response = requests.get(self.alpha_vantage_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Check for error messages
            if 'Error Message' in data:
                logger.error(f"API returned error: {data['Error Message']}")
                return None
            
            # Extract time series data
            time_series_key = next((k for k in data.keys() if 'Time Series' in k), None)
            if not time_series_key:
                logger.error("Time series data not found in response")
                return None
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            
            # Rename columns
            column_mapping = {}
            for col in df.columns:
                if 'open' in col.lower():
                    column_mapping[col] = 'open'
                elif 'high' in col.lower():
                    column_mapping[col] = 'high'
                elif 'low' in col.lower():
                    column_mapping[col] = 'low'
                elif 'close' in col.lower():
                    column_mapping[col] = 'close'
                elif 'volume' in col.lower():
                    column_mapping[col] = 'volume'
            
            df = df.rename(columns=column_mapping)
            
            # Convert to numeric values
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col])
            
            # Convert index to datetime
            df.index = pd.to_datetime(df.index)
            
            # Filter by date range
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            df = df[(df.index >= start) & (df.index <= end)]
            
            # Add symbol column
            df['symbol'] = symbol
            
            # Sort by date (ascending)
            df.sort_index(inplace=True)
            
            # Save to database if requested
            if save:
                self._save_to_database(df, symbol, interval)
            
            logger.info(f"Successfully fetched {len(df)} data points for {symbol} from Alpha Vantage")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol} from Alpha Vantage: {str(e)}")
            return None
    
    def _save_to_database(self, df, symbol, interval='1d'):
        """
        Save the fetched data to database.
        
        Args:
            df (pandas.DataFrame): DataFrame to save
            symbol (str): Stock symbol
            interval (str): Data interval
        """
        try:
            table_name = f"stock_{symbol.lower()}_{interval}"
            self.db_connector.save_dataframe(df, table_name)
            logger.info(f"Saved {len(df)} rows of {symbol} data to database")
        except Exception as e:
            logger.error(f"Error saving {symbol} data to database: {str(e)}")
    
    def load_from_database(self, symbol, start_date=None, end_date=None, interval='1d'):
        """
        Load stock data from database.
        
        Args:
            symbol (str): Stock symbol
            start_date (str, optional): Start date in format 'YYYY-MM-DD'
            end_date (str, optional): End date in format 'YYYY-MM-DD'
            interval (str, optional): Data interval ('1d', '1h', etc.)
            
        Returns:
            pandas.DataFrame: DataFrame containing the stock data
        """
        table_name = f"stock_{symbol.lower()}_{interval}"
        
        if not self.db_connector.table_exists(table_name):
            logger.warning(f"Table {table_name} does not exist in database")
            return None
        
        query = f"SELECT * FROM {table_name}"
        conditions = []
        
        if start_date:
            conditions.append(f"date >= '{start_date}'")
        if end_date:
            conditions.append(f"date <= '{end_date}'")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY date ASC"
        
        try:
            df = self.db_connector.run_query(query)
            
            if df.empty:
                logger.warning(f"No data found in database for {symbol}")
                return None
            
            # Convert date column to datetime and set as index
            if 'date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            # Make sure columns are lowercase
            df.columns = [col.lower() for col in df.columns]
            
            # Convert numeric columns
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col])
            
            logger.info(f"Loaded {len(df)} rows of {symbol} data from database")
            return df
            
        except Exception as e:
            logger.error(f"Error loading {symbol} data from database: {str(e)}")
            return None
    
    def fetch_multiple_stocks(self, symbols=None, start_date=None, end_date=None, provider=None, interval='1d', save=True):
        """
        Fetch data for multiple stocks.
        
        Args:
            symbols (list, optional): List of stock symbols, defaults to watchlist
            start_date (str, optional): Start date in format 'YYYY-MM-DD'
            end_date (str, optional): End date in format 'YYYY-MM-DD'
            provider (str, optional): Data provider ('yahoo', 'alpha_vantage')
            interval (str, optional): Data interval ('1d', '1h', etc.)
            save (bool): Whether to save the data to database
            
        Returns:
            dict: Dictionary of DataFrames keyed by symbol
        """
        if symbols is None:
            symbols = self.watchlist
        
        if not symbols:
            logger.warning("No symbols provided and watchlist is empty")
            return {}
        
        results = {}
        
        for i, symbol in enumerate(symbols):
            # Respect API rate limits for Alpha Vantage
            if provider == 'alpha_vantage' and i > 0 and i % self.call_limit == 0:
                logger.info(f"Reached API call limit. Pausing for 60 seconds...")
                time.sleep(60)
            
            df = self.fetch_stock_data(symbol, start_date, end_date, provider, interval, save)
            if df is not None and not df.empty:
                results[symbol] = df
        
        logger.info(f"Fetched data for {len(results)}/{len(symbols)} symbols")
        return results
    
    def calculate_returns(self, df, periods=[1, 5, 20, 60, 252]):
        """
        Calculate returns over different periods.
        
        Args:
            df (pandas.DataFrame): DataFrame with stock data
            periods (list): List of periods for return calculation
            
        Returns:
            pandas.DataFrame: DataFrame with added return columns
        """
        if df is None or df.empty or 'close' not in df.columns:
            logger.warning("Cannot calculate returns: Invalid data")
            return df
        
        result_df = df.copy()
        
        # Calculate simple returns
        for period in periods:
            column_name = f'return_{period}d'
            result_df[column_name] = result_df['close'].pct_change(period)
        
        # Calculate log returns
        for period in periods:
            column_name = f'log_return_{period}d'
            result_df[column_name] = np.log(result_df['close'] / result_df['close'].shift(period))
        
        # Calculate cumulative returns
        result_df['cum_return'] = (1 + result_df['return_1d']).cumprod() - 1
        
        return result_df
    
    def calculate_volatility(self, df, windows=[20, 60, 252]):
        """
        Calculate volatility over different windows.
        
        Args:
            df (pandas.DataFrame): DataFrame with stock data
            windows (list): List of windows for volatility calculation
            
        Returns:
            pandas.DataFrame: DataFrame with added volatility columns
        """
        if df is None or df.empty or 'return_1d' not in df.columns:
            logger.warning("Cannot calculate volatility: Missing daily returns")
            if df is not None and not df.empty and 'close' in df.columns:
                df = self.calculate_returns(df)
            else:
                return df
        
        result_df = df.copy()
        
        # Calculate rolling volatility
        for window in windows:
            column_name = f'volatility_{window}d'
            result_df[column_name] = result_df['return_1d'].rolling(window=window).std() * np.sqrt(252)
        
        return result_df
    
    def get_market_index_data(self, index_symbol='^GSPC', start_date=None, end_date=None, save=True):
        """
        Fetch market index data (e.g., S&P 500).
        
        Args:
            index_symbol (str): Index symbol (default: ^GSPC for S&P 500)
            start_date (str, optional): Start date in format 'YYYY-MM-DD'
            end_date (str, optional): End date in format 'YYYY-MM-DD'
            save (bool): Whether to save the data to database
            
        Returns:
            pandas.DataFrame: DataFrame containing the index data
        """
        return self.fetch_stock_data(index_symbol, start_date, end_date, 'yahoo', '1d', save)
    
    def calculate_beta(self, stock_df, market_df=None, window=252):
        """
        Calculate beta (market sensitivity) of a stock.
        
        Args:
            stock_df (pandas.DataFrame): DataFrame with stock data
            market_df (pandas.DataFrame, optional): DataFrame with market index data
            window (int): Rolling window for beta calculation
            
        Returns:
            pandas.DataFrame: DataFrame with added beta column
        """
        if stock_df is None or stock_df.empty:
            logger.warning("Cannot calculate beta: Invalid stock data")
            return stock_df
        
        if 'return_1d' not in stock_df.columns:
            stock_df = self.calculate_returns(stock_df)
        
        # If market data not provided, fetch S&P 500
        if market_df is None:
            market_df = self.get_market_index_data()
            
            if market_df is None or market_df.empty:
                logger.warning("Cannot calculate beta: Invalid market data")
                return stock_df
            
            if 'return_1d' not in market_df.columns:
                market_df = self.calculate_returns(market_df)
        
        result_df = stock_df.copy()
        
        # Align data
        aligned_stock = result_df['return_1d']
        aligned_market = market_df['return_1d']
        
        if aligned_stock.index.equals(aligned_market.index):
            # Calculate rolling covariance and market variance
            rolling_cov = aligned_stock.rolling(window=window).cov(aligned_market)
            rolling_var = aligned_market.rolling(window=window).var()
            
            # Calculate beta
            result_df['beta'] = rolling_cov / rolling_var
            
            # Calculate alpha (Jensen's alpha)
            risk_free_rate = self.config.get('analysis', {}).get('risk_free_rate', 0.0)
            daily_rf = (1 + risk_free_rate) ** (1/252) - 1
            
            result_df['alpha'] = result_df['return_1d'] - (daily_rf + result_df['beta'] * (aligned_market - daily_rf))
        else:
            # Reindex market data to match stock data
            common_index = result_df.index.intersection(market_df.index)
            aligned_stock = result_df.loc[common_index, 'return_1d']
            aligned_market = market_df.loc[common_index, 'return_1d']
            
            # Create temporary DataFrame for calculations
            temp_df = pd.DataFrame({'stock_return': aligned_stock, 'market_return': aligned_market})
            
            # Calculate rolling beta
            temp_df['rolling_cov'] = temp_df['stock_return'].rolling(window=window).cov(temp_df['market_return'])
            temp_df['rolling_var'] = temp_df['market_return'].rolling(window=window).var()
            temp_df['beta'] = temp_df['rolling_cov'] / temp_df['rolling_var']
            
            # Add beta to result DataFrame
            result_df.loc[common_index, 'beta'] = temp_df['beta']
        
        return result_df
    
    def get_company_info(self, symbol):
        """
        Get company information for a stock symbol.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            dict: Dictionary containing company information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract relevant information
            company_info = {
                'symbol': symbol,
                'name': info.get('shortName', ''),
                'industry': info.get('industry', ''),
                'sector': info.get('sector', ''),
                'market_cap': info.get('marketCap', None),
                'pe_ratio': info.get('trailingPE', None),
                'dividend_yield': info.get('dividendYield', None),
                'website': info.get('website', ''),
                'description': info.get('longBusinessSummary', '')
            }
            
            return company_info
            
        except Exception as e:
            logger.error(f"Error getting company info for {symbol}: {str(e)}")
            return None
    
    def get_earnings_dates(self, symbol, limit=4):
        """
        Get upcoming and past earnings dates for a stock.
        
        Args:
            symbol (str): Stock symbol
            limit (int): Number of past and future earnings dates to retrieve
            
        Returns:
            pandas.DataFrame: DataFrame containing earnings dates
        """
        try:
            ticker = yf.Ticker(symbol)
            earnings = ticker.earnings_dates
            
            if earnings is None or earnings.empty:
                logger.warning(f"No earnings dates found for {symbol}")
                return None
            
            # Limit the number of entries
            earnings = earnings.head(limit)
            
            return earnings
            
        except Exception as e:
            logger.error(f"Error getting earnings dates for {symbol}: {str(e)}")
            return None 