"""
MarketMind: Neural Prediction Engine for Tech Equities
A multi-factor LSTM model for tech stock prediction with economic indicator integration.
"""

import os
import logging
import yaml

# Package version
__version__ = '0.1.0'

# Setup logging
logger = logging.getLogger('marketmind')

# Load configuration file
def load_config():
    """Load configuration from the YAML file."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    
    if not os.path.exists(config_path):
        logger.warning(f"Configuration file not found at {config_path}. Using default configuration.")
        return {}
    
    with open(config_path, 'r') as config_file:
        try:
            config = yaml.safe_load(config_file)
            return config
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            return {}

# Create necessary directories if they don't exist
def create_directories():
    """Create necessary directories for the project."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    directories = [
        os.path.join(base_dir, 'data', 'raw'),
        os.path.join(base_dir, 'data', 'processed'),
        os.path.join(base_dir, 'data', 'models'),
        os.path.join(base_dir, 'logs')
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

# Initialize logging
def setup_logging(config=None):
    """Configure logging based on settings."""
    if config is None:
        config = {}
    
    log_level = config.get('logging', {}).get('level', 'INFO')
    log_file = config.get('logging', {}).get('file', 'logs/market_mind.log')
    max_size = config.get('logging', {}).get('max_size', 10485760)  # 10 MB
    backup_count = config.get('logging', {}).get('backup_count', 5)
    
    # Ensure log directory exists
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configure logger
    logger.setLevel(getattr(logging, log_level))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized with level {log_level}")

# Initialize the package
config = load_config()
setup_logging(config)
create_directories()

logger.info(f"MarketMind v{__version__} initialized") 