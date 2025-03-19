"""
Attention Layer

This module implements attention mechanisms for LSTM networks.
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import logging

logger = logging.getLogger('marketmind.models.attention')

class AttentionLayer(Layer):
    """
    Attention mechanism for LSTM networks.
    Implements Bahdanau attention.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the attention layer.
        """
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        """
        Build the attention layer.
        
        Args:
            input_shape (tuple): Shape of the input tensor
        """
        # Ensure we have a 3D tensor
        assert len(input_shape) == 3
        
        # Extract dimensions
        self.time_steps = input_shape[1]
        self.hidden_size = input_shape[2]
        
        # Weight matrix for the hidden state
        self.W = self.add_weight(
            name='attention_weight',
            shape=(self.hidden_size, 1),
            initializer='random_normal',
            trainable=True
        )
        
        # Bias term
        self.b = self.add_weight(
            name='attention_bias',
            shape=(self.time_steps, 1),
            initializer='zeros',
            trainable=True
        )
        
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        """
        Forward pass of the attention layer.
        
        Args:
            inputs (tf.Tensor): Input tensor, shape (batch_size, time_steps, hidden_size)
            
        Returns:
            tf.Tensor: Context vector, shape (batch_size, hidden_size)
        """
        # Calculate attention weights
        # e_t = tanh(W * h_t + b)
        e = K.tanh(K.dot(inputs, self.W) + self.b)
        
        # Calculate attention scores
        # a_t = softmax(e_t)
        a = K.softmax(e, axis=1)
        
        # Multiply weights with input and sum over the time dimension
        # c = sum(a_t * h_t)
        context = inputs * a
        context = K.sum(context, axis=1)
        
        return context
    
    def compute_output_shape(self, input_shape):
        """
        Compute output shape of the layer.
        
        Args:
            input_shape (tuple): Input shape
            
        Returns:
            tuple: Output shape
        """
        return (input_shape[0], input_shape[2])
    
    def get_config(self):
        """
        Get the config of the layer.
        
        Returns:
            dict: Config dictionary
        """
        config = super(AttentionLayer, self).get_config()
        return config


class LuongAttention(Layer):
    """
    Luong attention mechanism for LSTM networks.
    """
    
    def __init__(self, attention_dim=None, **kwargs):
        """
        Initialize the Luong attention layer.
        
        Args:
            attention_dim (int, optional): Dimension of the attention mechanism
        """
        self.attention_dim = attention_dim
        super(LuongAttention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        """
        Build the Luong attention layer.
        
        Args:
            input_shape (tuple): Shape of the input tensor
        """
        # Ensure we have a 3D tensor
        assert len(input_shape) == 3
        
        # Extract dimensions
        self.time_steps = input_shape[1]
        self.hidden_size = input_shape[2]
        
        # If attention_dim is not specified, use hidden_size
        if self.attention_dim is None:
            self.attention_dim = self.hidden_size
        
        # Weight matrix for general attention
        self.W = self.add_weight(
            name='attention_weight',
            shape=(self.hidden_size, self.hidden_size),
            initializer='glorot_uniform',
            trainable=True
        )
        
        super(LuongAttention, self).build(input_shape)
    
    def call(self, inputs):
        """
        Forward pass of the Luong attention layer.
        
        Args:
            inputs (tf.Tensor): Input tensor, shape (batch_size, time_steps, hidden_size)
            
        Returns:
            tf.Tensor: Context vector, shape (batch_size, hidden_size)
        """
        # Get the last hidden state (query vector)
        query = inputs[:, -1, :]
        query = K.expand_dims(query, axis=1)  # (batch_size, 1, hidden_size)
        
        # Calculate attention scores
        # score = h_t^T * W * h_s
        score = K.dot(query, self.W)  # (batch_size, 1, hidden_size)
        score = K.batch_dot(score, K.permute_dimensions(inputs, (0, 2, 1)))  # (batch_size, 1, time_steps)
        
        # Apply softmax to get attention weights
        attention_weights = K.softmax(score, axis=-1)  # (batch_size, 1, time_steps)
        
        # Compute the context vector
        context = K.batch_dot(attention_weights, inputs)  # (batch_size, 1, hidden_size)
        context = K.squeeze(context, axis=1)  # (batch_size, hidden_size)
        
        return context
    
    def compute_output_shape(self, input_shape):
        """
        Compute output shape of the layer.
        
        Args:
            input_shape (tuple): Input shape
            
        Returns:
            tuple: Output shape
        """
        return (input_shape[0], input_shape[2])
    
    def get_config(self):
        """
        Get the config of the layer.
        
        Returns:
            dict: Config dictionary
        """
        config = super(LuongAttention, self).get_config()
        config.update({
            'attention_dim': self.attention_dim
        })
        return config


def create_attention_lstm_model(input_shape, lstm_units, attention_type='bahdanau', dropout_rate=0.2):
    """
    Create an LSTM model with attention.
    
    Args:
        input_shape (tuple): Shape of input data
        lstm_units (list): List of units for LSTM layers
        attention_type (str): Type of attention mechanism ('bahdanau' or 'luong')
        dropout_rate (float): Dropout rate
        
    Returns:
        tf.keras.Model: LSTM model with attention
    """
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate
    
    inputs = Input(shape=input_shape)
    
    # Add LSTM layers
    lstm_output = inputs
    for i, units in enumerate(lstm_units[:-1]):
        lstm_output = LSTM(units, return_sequences=True, dropout=dropout_rate)(lstm_output)
    
    # Last LSTM layer should return sequences for attention
    lstm_output = LSTM(lstm_units[-1], return_sequences=True, dropout=dropout_rate)(lstm_output)
    
    # Apply attention
    if attention_type.lower() == 'bahdanau':
        context_vector = AttentionLayer()(lstm_output)
    elif attention_type.lower() == 'luong':
        context_vector = LuongAttention()(lstm_output)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")
    
    # Get the last output of the LSTM
    last_output = lstm_output[:, -1, :]
    
    # Concatenate context vector and last output
    attention_vector = Concatenate()([context_vector, last_output])
    
    # Add dropout
    attention_vector = Dropout(dropout_rate)(attention_vector)
    
    # Output layer
    output = Dense(1)(attention_vector)
    
    # Build model
    model = Model(inputs=inputs, outputs=output)
    
    return model 