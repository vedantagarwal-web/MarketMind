#!/usr/bin/env python
"""
MarketMind: Main script to run the stock prediction process.
"""

import os
import sys
import argparse
import logging
import pandas as pd
from datetime import datetime

# Add project directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from marketmind import load_config
from marketmind.data_acquisition.stock_data import StockDataFetcher

# Import optional modules
try:
    from marketmind.data_acquisition.news_data import NewsDataFetcher
    NEWS_AVAILABLE = True
except ImportError:
    NEWS_AVAILABLE = False
    logging.getLogger('marketmind.main').warning(
        "News data fetcher not available (missing dependencies: newspaper3k)"
    )

try:
    from marketmind.preprocessing.technical_indicators import TechnicalIndicator
    TECHNICAL_AVAILABLE = True
except ImportError:
    TECHNICAL_AVAILABLE = False
    logging.getLogger('marketmind.main').warning(
        "Technical indicators not available"
    )

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/marketmind.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('marketmind.main')

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='MarketMind: Stock Price Prediction System')
    
    parser.add_argument('--symbol', type=str, default='AAPL',
                        help='Stock symbol to predict (default: AAPL)')
    
    parser.add_argument('--start-date', type=str, default='2023-01-01',
                        help='Start date for historical data (default: 2023-01-01)')
    
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date for historical data (default: today)')
    
    parser.add_argument('--provider', type=str, default=None,
                        help='Data provider (yahoo or alpha_vantage, default: configured default)')
    
    parser.add_argument('--interval', type=str, default='1d',
                        help='Data interval (1d, 1h, etc., default: 1d)')
    
    parser.add_argument('--save', action='store_true',
                        help='Save data to database')
    
    parser.add_argument('--no-technical-indicators', action='store_true',
                        help='Disable technical indicators')
    
    parser.add_argument('--no-news', action='store_true',
                        help='Disable news sentiment analysis')
    
    parser.add_argument('--plot', action='store_true',
                        help='Generate and display plots')
    
    return parser.parse_args()

def main():
    """
    Main function to run the stock prediction process.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Set end date if not provided
    if args.end_date is None:
        args.end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Load configuration
    config = load_config()
    
    # Create directories if they don't exist
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    logger.info(f"Starting MarketMind for symbol: {args.symbol}")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    
    # Fetch stock data
    logger.info("Fetching stock data...")
    stock_fetcher = StockDataFetcher(config)
    stock_data = stock_fetcher.fetch_stock_data(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        provider=args.provider,
        interval=args.interval,
        save=args.save
    )
    
    if stock_data is None or stock_data.empty:
        logger.error("Failed to fetch stock data. Exiting.")
        return
    
    logger.info(f"Fetched {len(stock_data)} days of stock data")
    
    # Add technical indicators if enabled and available
    if not args.no_technical_indicators and TECHNICAL_AVAILABLE:
        logger.info("Adding technical indicators...")
        tech_indicator = TechnicalIndicator(config)
        stock_data = tech_indicator.add_all_indicators(stock_data, inplace=True)
        logger.info(f"Added technical indicators. Data shape: {stock_data.shape}")
    elif not args.no_technical_indicators and not TECHNICAL_AVAILABLE:
        logger.warning("Technical indicators module not available")
    
    # Add news sentiment if enabled and available
    if not args.no_news and NEWS_AVAILABLE:
        logger.info("Fetching news data for sentiment analysis...")
        news_fetcher = NewsDataFetcher(config)
        news_data = news_fetcher.fetch_company_news(
            company_symbols=[args.symbol],
            start_date=args.start_date,
            end_date=args.end_date,
            save=True
        )
        logger.info(f"Fetched news data for {args.symbol}")
    elif not args.no_news and not NEWS_AVAILABLE:
        logger.warning("News data fetcher not available")
    
    # Calculate basic statistics
    logger.info("Calculating basic statistics...")
    if 'close' in stock_data.columns:
        avg_price = stock_data['close'].mean()
        max_price = stock_data['close'].max()
        min_price = stock_data['close'].min()
        latest_price = stock_data['close'].iloc[-1]
        
        logger.info(f"Average price: {avg_price:.2f}")
        logger.info(f"Highest price: {max_price:.2f}")
        logger.info(f"Lowest price: {min_price:.2f}")
        logger.info(f"Latest price: {latest_price:.2f}")
    
    # Calculate returns
    if hasattr(stock_fetcher, 'calculate_returns'):
        logger.info("Calculating returns...")
        returns_data = stock_fetcher.calculate_returns(stock_data)
        
        if 'return_1d' in returns_data.columns:
            avg_return = returns_data['return_1d'].mean() * 100
            volatility = returns_data['return_1d'].std() * 100
            
            logger.info(f"Average daily return: {avg_return:.2f}%")
            logger.info(f"Volatility (std dev of daily returns): {volatility:.2f}%")
    
    # Display sample data
    logger.info("Sample data:")
    logger.info(stock_data.head())
    
    # Generate plots if enabled
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            
            logger.info("Generating plots...")
            plt.figure(figsize=(14, 7))
            plt.plot(stock_data.index, stock_data['close'])
            plt.title(f"{args.symbol} Stock Price ({args.start_date} to {args.end_date})")
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.grid(True)
            plt.tight_layout()
            
            # Save the plot
            plot_path = f"data/plots/{args.symbol}_{args.start_date}_{args.end_date}.png"
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path)
            logger.info(f"Plot saved to {plot_path}")
            
            plt.show()
        except ImportError:
            logger.warning("Matplotlib not available. Skipping plots.")
    
    logger.info("MarketMind process completed successfully")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")
        sys.exit(1) 