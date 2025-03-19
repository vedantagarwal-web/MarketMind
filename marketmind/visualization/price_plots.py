"""
Price Plots

This module provides visualization tools for stock price predictions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from datetime import datetime, timedelta

logger = logging.getLogger('marketmind.visualization.price_plots')

class PricePlotter:
    """
    Handles visualization of stock price predictions.
    """
    
    def __init__(self, config=None):
        """
        Initialize the PricePlotter with configuration.
        
        Args:
            config (dict): Configuration dictionary containing visualization settings.
        """
        from .. import load_config
        self.config = config or load_config()
        self.theme = self.config.get('visualization', {}).get('theme', 'dark')
        self.figures = []
    
    def plot_prediction_vs_actual(self, actual_df, prediction_df, title=None, use_plotly=True):
        """
        Plot predicted vs actual prices.
        
        Args:
            actual_df (pandas.DataFrame): DataFrame with actual prices
            prediction_df (pandas.DataFrame): DataFrame with predicted prices
            title (str, optional): Plot title
            use_plotly (bool): Whether to use Plotly (True) or Matplotlib (False)
            
        Returns:
            object: Plot figure
        """
        if use_plotly:
            return self._plot_prediction_vs_actual_plotly(actual_df, prediction_df, title)
        else:
            return self._plot_prediction_vs_actual_mpl(actual_df, prediction_df, title)
    
    def _plot_prediction_vs_actual_plotly(self, actual_df, prediction_df, title=None):
        """
        Plot predicted vs actual prices using Plotly.
        
        Args:
            actual_df (pandas.DataFrame): DataFrame with actual prices
            prediction_df (pandas.DataFrame): DataFrame with predicted prices
            title (str, optional): Plot title
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure
        """
        # Ensure we have matching indices
        common_index = actual_df.index.intersection(prediction_df.index)
        
        if len(common_index) == 0:
            logger.warning("No common dates between actual and prediction data")
            return None
        
        # Extract data
        actual = actual_df.loc[common_index]['close']
        predicted = prediction_df.loc[common_index]['prediction']
        
        # Create figure
        fig = go.Figure()
        
        # Add traces
        fig.add_trace(
            go.Scatter(
                x=common_index,
                y=actual,
                mode='lines',
                name='Actual',
                line=dict(color='#1f77b4', width=2)
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=common_index,
                y=predicted,
                mode='lines',
                name='Predicted',
                line=dict(color='#ff7f0e', width=2)
            )
        )
        
        # Calculate error
        error = np.abs(actual - predicted)
        
        # Add error band
        fig.add_trace(
            go.Scatter(
                x=common_index,
                y=predicted + error,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=common_index,
                y=predicted - error,
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(255, 127, 14, 0.3)',
                fill='tonexty',
                name='Error Band'
            )
        )
        
        # Update layout
        fig.update_layout(
            title=title or 'Predicted vs Actual Prices',
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified',
            template='plotly_dark' if self.theme == 'dark' else 'plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Store figure
        self.figures.append(fig)
        
        return fig
    
    def _plot_prediction_vs_actual_mpl(self, actual_df, prediction_df, title=None):
        """
        Plot predicted vs actual prices using Matplotlib.
        
        Args:
            actual_df (pandas.DataFrame): DataFrame with actual prices
            prediction_df (pandas.DataFrame): DataFrame with predicted prices
            title (str, optional): Plot title
            
        Returns:
            matplotlib.figure.Figure: Matplotlib figure
        """
        # Ensure we have matching indices
        common_index = actual_df.index.intersection(prediction_df.index)
        
        if len(common_index) == 0:
            logger.warning("No common dates between actual and prediction data")
            return None
        
        # Extract data
        actual = actual_df.loc[common_index]['close']
        predicted = prediction_df.loc[common_index]['prediction']
        
        # Set style
        plt.style.use('dark_background' if self.theme == 'dark' else 'default')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot data
        ax.plot(common_index, actual, label='Actual', color='#1f77b4', linewidth=2)
        ax.plot(common_index, predicted, label='Predicted', color='#ff7f0e', linewidth=2)
        
        # Calculate error
        error = np.abs(actual - predicted)
        
        # Add error band
        ax.fill_between(
            common_index,
            predicted - error,
            predicted + error,
            color='#ff7f0e',
            alpha=0.3,
            label='Error Band'
        )
        
        # Add labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title(title or 'Predicted vs Actual Prices')
        ax.legend()
        
        # Format x-axis date labels
        fig.autofmt_xdate()
        
        # Adjust layout
        plt.tight_layout()
        
        # Store figure
        self.figures.append(fig)
        
        return fig
    
    def plot_future_prediction(self, historical_df, future_df, title=None, use_plotly=True):
        """
        Plot future price predictions with historical data.
        
        Args:
            historical_df (pandas.DataFrame): DataFrame with historical prices
            future_df (pandas.DataFrame): DataFrame with future predictions
            title (str, optional): Plot title
            use_plotly (bool): Whether to use Plotly (True) or Matplotlib (False)
            
        Returns:
            object: Plot figure
        """
        if use_plotly:
            return self._plot_future_prediction_plotly(historical_df, future_df, title)
        else:
            return self._plot_future_prediction_mpl(historical_df, future_df, title)
    
    def _plot_future_prediction_plotly(self, historical_df, future_df, title=None):
        """
        Plot future price predictions with historical data using Plotly.
        
        Args:
            historical_df (pandas.DataFrame): DataFrame with historical prices
            future_df (pandas.DataFrame): DataFrame with future predictions
            title (str, optional): Plot title
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure
        """
        # Create figure
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(
            go.Scatter(
                x=historical_df.index,
                y=historical_df['close'],
                mode='lines',
                name='Historical',
                line=dict(color='#1f77b4', width=2)
            )
        )
        
        # Add future prediction
        fig.add_trace(
            go.Scatter(
                x=future_df.index,
                y=future_df['prediction'],
                mode='lines',
                name='Prediction',
                line=dict(color='#ff7f0e', width=2)
            )
        )
        
        # Add confidence intervals (assuming 15% uncertainty for demonstration)
        uncertainty = 0.15 * future_df['prediction']
        
        fig.add_trace(
            go.Scatter(
                x=future_df.index,
                y=future_df['prediction'] + uncertainty,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=future_df.index,
                y=future_df['prediction'] - uncertainty,
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(255, 127, 14, 0.3)',
                fill='tonexty',
                name='Confidence Interval'
            )
        )
        
        # Add vertical line to separate historical and prediction
        if not historical_df.empty and not future_df.empty:
            last_historical_date = historical_df.index[-1]
            
            fig.add_shape(
                type="line",
                x0=last_historical_date,
                y0=0,
                x1=last_historical_date,
                y1=1,
                yref="paper",
                line=dict(
                    color="gray",
                    width=2,
                    dash="dash",
                )
            )
            
            # Add annotation
            fig.add_annotation(
                x=last_historical_date,
                y=1,
                yref="paper",
                text="Prediction Start",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40
            )
        
        # Update layout
        fig.update_layout(
            title=title or 'Future Price Prediction',
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified',
            template='plotly_dark' if self.theme == 'dark' else 'plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Store figure
        self.figures.append(fig)
        
        return fig
    
    def _plot_future_prediction_mpl(self, historical_df, future_df, title=None):
        """
        Plot future price predictions with historical data using Matplotlib.
        
        Args:
            historical_df (pandas.DataFrame): DataFrame with historical prices
            future_df (pandas.DataFrame): DataFrame with future predictions
            title (str, optional): Plot title
            
        Returns:
            matplotlib.figure.Figure: Matplotlib figure
        """
        # Set style
        plt.style.use('dark_background' if self.theme == 'dark' else 'default')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical data
        ax.plot(
            historical_df.index,
            historical_df['close'],
            label='Historical',
            color='#1f77b4',
            linewidth=2
        )
        
        # Plot future prediction
        ax.plot(
            future_df.index,
            future_df['prediction'],
            label='Prediction',
            color='#ff7f0e',
            linewidth=2
        )
        
        # Add confidence intervals (assuming 15% uncertainty for demonstration)
        uncertainty = 0.15 * future_df['prediction']
        ax.fill_between(
            future_df.index,
            future_df['prediction'] - uncertainty,
            future_df['prediction'] + uncertainty,
            color='#ff7f0e',
            alpha=0.3,
            label='Confidence Interval'
        )
        
        # Add vertical line to separate historical and prediction
        if not historical_df.empty and not future_df.empty:
            last_historical_date = historical_df.index[-1]
            ax.axvline(
                x=last_historical_date,
                color='gray',
                linestyle='--',
                label='Prediction Start'
            )
        
        # Add labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title(title or 'Future Price Prediction')
        ax.legend()
        
        # Format x-axis date labels
        fig.autofmt_xdate()
        
        # Adjust layout
        plt.tight_layout()
        
        # Store figure
        self.figures.append(fig)
        
        return fig
    
    def plot_feature_importance(self, feature_names, importance_values, title=None, use_plotly=True):
        """
        Plot feature importance.
        
        Args:
            feature_names (list): List of feature names
            importance_values (list): List of importance values
            title (str, optional): Plot title
            use_plotly (bool): Whether to use Plotly (True) or Matplotlib (False)
            
        Returns:
            object: Plot figure
        """
        if use_plotly:
            return self._plot_feature_importance_plotly(feature_names, importance_values, title)
        else:
            return self._plot_feature_importance_mpl(feature_names, importance_values, title)
    
    def _plot_feature_importance_plotly(self, feature_names, importance_values, title=None):
        """
        Plot feature importance using Plotly.
        
        Args:
            feature_names (list): List of feature names
            importance_values (list): List of importance values
            title (str, optional): Plot title
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure
        """
        # Sort by importance
        sorted_idx = np.argsort(importance_values)
        feature_names = [feature_names[i] for i in sorted_idx]
        importance_values = [importance_values[i] for i in sorted_idx]
        
        # Create figure
        fig = go.Figure(
            go.Bar(
                x=importance_values,
                y=feature_names,
                orientation='h'
            )
        )
        
        # Update layout
        fig.update_layout(
            title=title or 'Feature Importance',
            xaxis_title='Importance',
            yaxis_title='Feature',
            template='plotly_dark' if self.theme == 'dark' else 'plotly_white'
        )
        
        # Store figure
        self.figures.append(fig)
        
        return fig
    
    def _plot_feature_importance_mpl(self, feature_names, importance_values, title=None):
        """
        Plot feature importance using Matplotlib.
        
        Args:
            feature_names (list): List of feature names
            importance_values (list): List of importance values
            title (str, optional): Plot title
            
        Returns:
            matplotlib.figure.Figure: Matplotlib figure
        """
        # Set style
        plt.style.use('dark_background' if self.theme == 'dark' else 'default')
        
        # Sort by importance
        sorted_idx = np.argsort(importance_values)
        feature_names = [feature_names[i] for i in sorted_idx]
        importance_values = [importance_values[i] for i in sorted_idx]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot data
        ax.barh(feature_names, importance_values)
        
        # Add labels and title
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        ax.set_title(title or 'Feature Importance')
        
        # Adjust layout
        plt.tight_layout()
        
        # Store figure
        self.figures.append(fig)
        
        return fig
    
    def save_all_figures(self, directory='plots', format='png'):
        """
        Save all figures to disk.
        
        Args:
            directory (str): Directory to save figures
            format (str): File format ('png', 'html', 'pdf', etc.)
            
        Returns:
            list: List of saved file paths
        """
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        saved_paths = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for i, fig in enumerate(self.figures):
            # Generate filename
            filename = f"plot_{timestamp}_{i}.{format}"
            path = os.path.join(directory, filename)
            
            try:
                # Check if it's a Plotly figure
                if hasattr(fig, 'write_image'):
                    if format == 'html':
                        fig.write_html(path)
                    else:
                        fig.write_image(path)
                # Matplotlib figure
                else:
                    fig.savefig(path)
                
                saved_paths.append(path)
                logger.info(f"Saved figure to {path}")
                
            except Exception as e:
                logger.error(f"Error saving figure to {path}: {str(e)}")
        
        return saved_paths
    
    def show(self):
        """
        Display all figures.
        """
        for fig in self.figures:
            # Check if it's a Plotly figure
            if hasattr(fig, 'show'):
                fig.show()
            # Matplotlib figure
            else:
                plt.figure(fig.number)
                plt.show()
    
    def clear(self):
        """
        Clear all figures.
        """
        self.figures = [] 