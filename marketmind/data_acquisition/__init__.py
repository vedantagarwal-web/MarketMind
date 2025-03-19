"""
Data Acquisition Package

This package contains modules for fetching various types of market data.
"""

import logging

logger = logging.getLogger('marketmind.data_acquisition')

# Import only stock data fetcher directly
from .stock_data import StockDataFetcher

# Other modules are imported on demand to avoid unnecessary dependencies
# and allow individual module testing
try:
    from .economic_data import EconomicDataFetcher
    __all__ = ['StockDataFetcher', 'EconomicDataFetcher']
except ImportError:
    logger.warning("Economic data fetcher not available")
    __all__ = ['StockDataFetcher']

# Try to import NewsDataFetcher, but don't fail if dependencies are missing
try:
    from .news_data import NewsDataFetcher
    __all__ = __all__ + ['NewsDataFetcher']
except ImportError:
    logger.warning("News data fetcher not available (missing dependencies: newspaper3k)") 