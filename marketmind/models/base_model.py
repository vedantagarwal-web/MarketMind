"""
Base Model

This module defines the base class for all models.
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime

logger = logging.getLogger('marketmind.models.base_model')

class BaseModel(ABC):
    """
    Abstract base class for all models.
    """
    
    def __init__(self, config=None):
        """
        Initialize the BaseModel with configuration.
        
        Args:
            config (dict): Configuration dictionary containing model settings.
        """
        from .. import load_config
        self.config = config or load_config()
        self.model_config = self.config.get('model', {})
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.target_column = None
        self.sequence_length = self.model_config.get('sequence_length', 60)
        self.forecast_horizon = self.model_config.get('forecast_horizon', 7)
        self.train_test_split = self.model_config.get('train_test_split', 0.8)
        self.validation_split = self.model_config.get('validation_split', 0.2)
        self.model_path = None
    
    @abstractmethod
    def build_model(self, input_shape):
        """
        Build the model architecture.
        
        Args:
            input_shape (tuple): Shape of input data
            
        Returns:
            model: The constructed model
        """
        pass
    
    @abstractmethod
    def preprocess_data(self, df):
        """
        Preprocess the data for model training.
        
        Args:
            df (pandas.DataFrame): Raw data
            
        Returns:
            tuple: Processed X and y data
        """
        pass
    
    @abstractmethod
    def train(self, df, target_column='close', feature_columns=None):
        """
        Train the model on the given data.
        
        Args:
            df (pandas.DataFrame): Training data
            target_column (str): Column to predict
            feature_columns (list): Columns to use as features
            
        Returns:
            object: Trained model
        """
        pass
    
    @abstractmethod
    def predict(self, df=None, days=None):
        """
        Make predictions using the trained model.
        
        Args:
            df (pandas.DataFrame, optional): Data to predict on
            days (int, optional): Number of days to predict
            
        Returns:
            pandas.DataFrame: Predictions
        """
        pass
    
    def save_model(self, file_path=None):
        """
        Save the model to disk.
        
        Args:
            file_path (str, optional): Path to save the model
            
        Returns:
            str: Path where model was saved
        """
        if self.model is None:
            logger.error("Cannot save model: Model not trained")
            return None
        
        if file_path is None:
            # Generate a default path
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = self.__class__.__name__.lower()
            
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            data_dir = os.path.join(base_dir, 'data', 'models')
            os.makedirs(data_dir, exist_ok=True)
            
            file_path = os.path.join(data_dir, f"{model_name}_{timestamp}.pkl")
        
        try:
            # Create a dictionary with all necessary components to restore the model
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'target_column': self.target_column,
                'sequence_length': self.sequence_length,
                'forecast_horizon': self.forecast_horizon,
                'config': self.config,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'model_type': self.__class__.__name__
                }
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {file_path}")
            self.model_path = file_path
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return None
    
    def load_model(self, file_path):
        """
        Load a model from disk.
        
        Args:
            file_path (str): Path to the saved model
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.target_column = model_data['target_column']
            self.sequence_length = model_data['sequence_length']
            self.forecast_horizon = model_data['forecast_horizon']
            self.config = model_data['config']
            self.model_path = file_path
            
            logger.info(f"Model loaded from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def _create_sequences(self, data, sequence_length, target_idx=-1):
        """
        Create sequences for time series prediction.
        
        Args:
            data (numpy.ndarray): Input data
            sequence_length (int): Length of sequences
            target_idx (int): Index of target column
            
        Returns:
            tuple: X and y sequences
        """
        X, y = [], []
        
        for i in range(len(data) - sequence_length - self.forecast_horizon + 1):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length + self.forecast_horizon - 1, target_idx])
        
        return np.array(X), np.array(y)
    
    def _create_multistep_sequences(self, data, sequence_length, forecast_horizon, target_idx=-1):
        """
        Create sequences for multi-step time series prediction.
        
        Args:
            data (numpy.ndarray): Input data
            sequence_length (int): Length of sequences
            forecast_horizon (int): Number of steps to forecast
            target_idx (int): Index of target column
            
        Returns:
            tuple: X and y sequences
        """
        X, y = [], []
        
        for i in range(len(data) - sequence_length - forecast_horizon + 1):
            X.append(data[i:(i + sequence_length)])
            y.append(data[(i + sequence_length):(i + sequence_length + forecast_horizon), target_idx])
        
        return np.array(X), np.array(y)
    
    def evaluate(self, test_df, metrics=None):
        """
        Evaluate the model on test data.
        
        Args:
            test_df (pandas.DataFrame): Test data
            metrics (list, optional): List of metrics to calculate
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            logger.error("Cannot evaluate model: Model not trained")
            return None
        
        if metrics is None:
            metrics = self.config.get('evaluation', {}).get('metrics', ['mse', 'mae', 'mape', 'r2'])
        
        # Make predictions
        predictions = self.predict(test_df)
        
        if predictions is None:
            logger.error("Cannot evaluate model: Prediction failed")
            return None
        
        # Extract actual values
        actual = test_df[self.target_column].values
        
        # Match length if needed
        min_length = min(len(actual), len(predictions))
        actual = actual[-min_length:]
        predicted = predictions.values[-min_length:]
        
        result = {}
        
        # Calculate metrics
        for metric in metrics:
            if metric.lower() == 'mse':
                from sklearn.metrics import mean_squared_error
                result['mse'] = mean_squared_error(actual, predicted)
            
            elif metric.lower() == 'mae':
                from sklearn.metrics import mean_absolute_error
                result['mae'] = mean_absolute_error(actual, predicted)
            
            elif metric.lower() == 'mape':
                from sklearn.metrics import mean_absolute_percentage_error
                result['mape'] = mean_absolute_percentage_error(actual, predicted)
            
            elif metric.lower() == 'r2':
                from sklearn.metrics import r2_score
                result['r2'] = r2_score(actual, predicted)
            
            elif metric.lower() == 'sharpe_ratio':
                # Calculate daily returns
                returns = np.diff(predicted) / predicted[:-1]
                result['sharpe_ratio'] = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
            
            elif metric.lower() == 'max_drawdown':
                # Calculate maximum drawdown
                cumulative = np.maximum.accumulate(predicted)
                drawdown = (cumulative - predicted) / cumulative
                result['max_drawdown'] = np.max(drawdown)
            
            else:
                logger.warning(f"Unknown metric: {metric}")
        
        return result 