#!/usr/bin/env python3
"""
Test script for the StockDataFetcher module.
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# Ensure the marketmind package is in the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from marketmind.data_acquisition.stock_data import StockDataFetcher, YAHOO_AVAILABLE
    print("Successfully imported StockDataFetcher")
    
    # Initialize the fetcher
    fetcher = StockDataFetcher()
    print("StockDataFetcher initialized")
    
    # Test data parameters
    symbol = 'AAPL'
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    print(f"Fetching {symbol} data from {start_date} to {end_date}")
    
    # Try all available data sources
    for provider in ['yahoo', 'alpha_vantage']:
        if provider == 'yahoo' and not YAHOO_AVAILABLE:
            print("Skipping Yahoo Finance (not available)")
            continue
            
        print(f"\nTrying {provider} data source...")
        data = fetcher.fetch_stock_data(symbol, start_date=start_date, end_date=end_date, 
                                       provider=provider, save=False)
        
        if data is not None and not data.empty:
            print(f"Successfully fetched {len(data)} data points for {symbol} from {provider}")
            print("\nSample data:")
            print(data.head())
            
            # Calculate basic statistics
            print("\nBasic statistics:")
            if 'close' in data.columns:
                print(f"Average closing price: {data['close'].mean():.2f}")
                print(f"Highest closing price: {data['close'].max():.2f}")
                print(f"Lowest closing price: {data['close'].min():.2f}")
            
            # Calculate returns
            if hasattr(fetcher, 'calculate_returns'):
                returns_data = fetcher.calculate_returns(data)
                print("\nReturns data:")
                print(returns_data[['close', 'return_1d']].head())
                
            # Stop after first successful fetch
            break
        else:
            print(f"Failed to fetch data from {provider}")
    else:
        print("\nAll data sources failed")
    
except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Error: {e}") 