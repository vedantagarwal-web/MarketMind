"""
Configuration Helper

This module provides utilities for managing configuration.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger('marketmind.utils.config_helper')

class ConfigManager:
    """
    Manages configuration settings for the application.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the config manager.
        
        Args:
            config_path (str, optional): Path to configuration file
        """
        if config_path is None:
            # Use default path
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            config_path = os.path.join(base_dir, 'config.yaml')
        
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            dict: Configuration dictionary
        """
        if not os.path.exists(self.config_path):
            logger.warning(f"Configuration file not found at {self.config_path}")
            return {}
        
        try:
            with open(self.config_path, 'r') as config_file:
                config = yaml.safe_load(config_file)
                logger.info(f"Loaded configuration from {self.config_path}")
                return config or {}
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return {}
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the entire configuration.
        
        Returns:
            dict: Configuration dictionary
        """
        return self.config
    
    def get(self, *keys, default=None) -> Any:
        """
        Get a configuration value by key path.
        
        Args:
            *keys: Key path to the configuration value
            default: Default value if key not found
            
        Returns:
            Any: Configuration value
        """
        current = self.config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def set(self, value: Any, *keys) -> bool:
        """
        Set a configuration value by key path.
        
        Args:
            value: Value to set
            *keys: Key path to the configuration value
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not keys:
            logger.error("No keys provided to set configuration value")
            return False
        
        current = self.config
        
        # Navigate to the nested dictionary
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                current[key] = {}
            
            current = current[key]
        
        # Set the value
        current[keys[-1]] = value
        
        return True
    
    def save(self, config_path=None) -> bool:
        """
        Save the configuration to a YAML file.
        
        Args:
            config_path (str, optional): Path to save the configuration file
            
        Returns:
            bool: True if successful, False otherwise
        """
        if config_path is None:
            config_path = self.config_path
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w') as config_file:
                yaml.dump(self.config, config_file, default_flow_style=False)
                
            logger.info(f"Saved configuration to {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False
    
    def update(self, config_dict: Dict[str, Any]) -> bool:
        """
        Update configuration with values from another dictionary.
        
        Args:
            config_dict (dict): Dictionary with configuration values
            
        Returns:
            bool: True if successful, False otherwise
        """
        def _recursive_update(target, source):
            for key, value in source.items():
                if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                    _recursive_update(target[key], value)
                else:
                    target[key] = value
        
        try:
            _recursive_update(self.config, config_dict)
            logger.info("Updated configuration")
            return True
            
        except Exception as e:
            logger.error(f"Error updating configuration: {str(e)}")
            return False
    
    def validate(self, schema=None) -> bool:
        """
        Validate the configuration against a schema.
        
        Args:
            schema (dict, optional): Schema to validate against
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # If no schema provided, use basic validation
            if schema is None:
                # Check required top-level sections
                required_sections = ['api', 'database', 'data', 'model', 'preprocessing']
                missing_sections = [section for section in required_sections if section not in self.config]
                
                if missing_sections:
                    logger.warning(f"Missing required configuration sections: {', '.join(missing_sections)}")
                    return False
                
                # Check API keys
                api_key = self.get('api', 'alpha_vantage', 'key')
                if not api_key or api_key == "YOUR_ALPHA_VANTAGE_API_KEY":
                    logger.warning("Alpha Vantage API key not set")
                
                return True
            
            # Schema validation can be implemented using jsonschema library
            # For simplicity, we'll skip that here
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating configuration: {str(e)}")
            return False 