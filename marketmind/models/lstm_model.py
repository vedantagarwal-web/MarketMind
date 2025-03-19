"""
LSTM Model

This module implements an LSTM model for stock price prediction.
"""

import numpy as np
import pandas as pd
import logging
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from .base_model import BaseModel

logger = logging.getLogger('marketmind.models.lstm_model')

class LSTMModel(BaseModel):
    """
    LSTM model for stock price prediction.
    """
    
    def __init__(self, config=None):
        """
        Initialize the LSTM model with configuration.
        
        Args:
            config (dict): Configuration dictionary containing model settings.
        """
        super().__init__(config)
        self.lstm_config = self.model_config.get('lstm', {})
        self.architecture = self.lstm_config.get('architecture', [128, 64, 32])
        self.dropout_rate = self.lstm_config.get('dropout_rate', 0.2)
        self.recurrent_dropout_rate = self.lstm_config.get('recurrent_dropout_rate', 0.2)
        self.activation = self.lstm_config.get('activation', 'relu')
        self.recurrent_activation = self.lstm_config.get('recurrent_activation', 'sigmoid')
        self.optimizer = self.lstm_config.get('optimizer', 'adam')
        self.loss = self.lstm_config.get('loss', 'mse')
        self.batch_size = self.lstm_config.get('batch_size', 32)
        self.epochs = self.lstm_config.get('epochs', 100)
        self.early_stopping_patience = self.lstm_config.get('early_stopping_patience', 10)
        
        # Initialize TensorFlow session
        try:
            # Set GPU memory growth to avoid allocating all memory
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Using GPU: {len(gpus)} available")
            else:
                logger.info("No GPU devices found. Using CPU.")
        except Exception as e:
            logger.warning(f"Error setting GPU memory growth: {str(e)}")
    
    def build_model(self, input_shape):
        """
        Build the LSTM model architecture.
        
        Args:
            input_shape (tuple): Shape of input data (sequence_length, n_features)
            
        Returns:
            tensorflow.keras.models.Sequential: The constructed model
        """
        model = Sequential()
        
        # Add LSTM layers
        for i, units in enumerate(self.architecture):
            return_sequences = i < len(self.architecture) - 1
            
            if i == 0:
                # First layer needs input_shape
                model.add(LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    activation=self.activation,
                    recurrent_activation=self.recurrent_activation,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.recurrent_dropout_rate,
                    input_shape=input_shape
                ))
            else:
                # Subsequent layers
                model.add(LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    activation=self.activation,
                    recurrent_activation=self.recurrent_activation,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.recurrent_dropout_rate
                ))
            
            # Add dropout after each LSTM layer
            model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile model
        model.compile(optimizer=self.optimizer, loss=self.loss)
        
        # Log model summary
        model.summary(print_fn=lambda x: logger.debug(x))
        
        return model
    
    def preprocess_data(self, df):
        """
        Preprocess the data for LSTM model.
        
        Args:
            df (pandas.DataFrame): Raw data
            
        Returns:
            tuple: Processed X and y data, scaler
        """
        # Ensure there are no missing values
        df = df.copy()
        df = df.dropna()
        
        # Extract features and target
        features = df[self.feature_columns].values
        target = df[[self.target_column]].values
        
        # Scale features
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Scale target separately to preserve the relationship
        target_scaler = MinMaxScaler()
        scaled_target = target_scaler.fit_transform(target)
        
        # Combine scaled features and target for sequence creation
        scaled_data = np.hstack((scaled_features, scaled_target))
        
        # Create sequences
        target_idx = scaled_data.shape[1] - 1
        X, y = self._create_sequences(scaled_data, self.sequence_length, target_idx)
        
        return X, y, scaler, target_scaler
    
    def train(self, df, target_column='close', feature_columns=None):
        """
        Train the LSTM model on the given data.
        
        Args:
            df (pandas.DataFrame): Training data
            target_column (str): Column to predict
            feature_columns (list): Columns to use as features
            
        Returns:
            tensorflow.keras.models.Sequential: Trained model
        """
        if feature_columns is None:
            # Use all numeric columns except the target
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            feature_columns = [col for col in numeric_cols if col != target_column]
        
        # Store columns for future reference
        self.feature_columns = feature_columns
        self.target_column = target_column
        
        logger.info(f"Training LSTM model with {len(feature_columns)} features")
        logger.info(f"Features: {feature_columns}")
        logger.info(f"Target: {target_column}")
        
        # Preprocess data
        X, y, self.scaler, self.target_scaler = self.preprocess_data(df)
        
        # Split data into train and validation sets
        train_size = int(len(X) * (1 - self.validation_split))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Validation data shape: {X_val.shape}")
        
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.build_model(input_shape)
        
        # Create callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Add model checkpoint if we're saving to disk
        checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                       'data', 'models', 'checkpoints')
        os.makedirs(checkpoint_path, exist_ok=True)
        callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(checkpoint_path, 'lstm_model_checkpoint.h5'),
                save_best_only=True,
                monitor='val_loss'
            )
        )
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Get best epoch
        best_epoch = np.argmin(history.history['val_loss']) + 1
        best_val_loss = min(history.history['val_loss'])
        logger.info(f"Training completed. Best epoch: {best_epoch}, Best validation loss: {best_val_loss:.6f}")
        
        return self.model
    
    def predict(self, df=None, days=None):
        """
        Make predictions using the trained LSTM model.
        
        Args:
            df (pandas.DataFrame, optional): Data to predict on
            days (int, optional): Number of days to predict into the future
            
        Returns:
            pandas.DataFrame: Predictions
        """
        if self.model is None:
            logger.error("Cannot make predictions: Model not trained")
            return None
        
        if df is None and days is None:
            logger.error("Either df or days must be provided")
            return None
        
        # If df is provided, predict on that data
        if df is not None:
            # Ensure all required columns are present
            missing_columns = [col for col in self.feature_columns + [self.target_column] if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing columns in input data: {missing_columns}")
                return None
            
            # Preprocess data
            X, _, _, _ = self.preprocess_data(df)
            
            # Make predictions
            scaled_predictions = self.model.predict(X)
            
            # Inverse transform predictions
            predictions = self.target_scaler.inverse_transform(scaled_predictions)
            
            # Create results dataframe
            result = pd.DataFrame(index=df.index[self.sequence_length:], columns=['prediction'])
            result['prediction'] = predictions.flatten()
            
            return result
        
        # If days is provided, predict future values
        else:
            if not hasattr(self, 'last_data'):
                logger.error("Cannot predict future values: No previous data available")
                return None
            
            # Start with the last sequence
            current_sequence = self.last_data.copy()
            
            # Predict one step at a time
            future_predictions = []
            future_dates = []
            
            last_date = self.last_date
            for i in range(days):
                # Reshape for prediction
                X = current_sequence.reshape(1, self.sequence_length, current_sequence.shape[1])
                
                # Predict next value
                scaled_prediction = self.model.predict(X)
                
                # Create a new data point with the prediction
                new_point = current_sequence[-1].copy()
                new_point[-1] = scaled_prediction[0, 0]
                
                # Update sequence by removing first point and adding the new point
                current_sequence = np.vstack((current_sequence[1:], new_point))
                
                # Store the prediction
                inverse_prediction = self.target_scaler.inverse_transform(scaled_prediction)[0, 0]
                future_predictions.append(inverse_prediction)
                
                # Calculate next date
                last_date = last_date + timedelta(days=1)
                future_dates.append(last_date)
            
            # Create results dataframe
            result = pd.DataFrame(index=future_dates, columns=['prediction'])
            result['prediction'] = future_predictions
            
            return result
    
    def prepare_future_prediction(self, df):
        """
        Prepare the model for future predictions.
        
        Args:
            df (pandas.DataFrame): Historical data to base predictions on
            
        Returns:
            bool: True if preparation succeeded, False otherwise
        """
        if self.model is None:
            logger.error("Cannot prepare future prediction: Model not trained")
            return False
        
        # Ensure all required columns are present
        missing_columns = [col for col in self.feature_columns + [self.target_column] if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing columns in input data: {missing_columns}")
            return False
        
        # Get the last sequence
        features = df[self.feature_columns].values
        target = df[[self.target_column]].values
        
        # Scale features
        scaled_features = self.scaler.transform(features)
        
        # Scale target
        scaled_target = self.target_scaler.transform(target)
        
        # Combine scaled features and target
        scaled_data = np.hstack((scaled_features, scaled_target))
        
        # Store the last sequence
        self.last_data = scaled_data[-self.sequence_length:]
        self.last_date = df.index[-1]
        
        return True 