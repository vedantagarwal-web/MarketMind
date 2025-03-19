"""
Sentiment Analysis Module

This module handles sentiment analysis of news headlines and social media posts
related to stocks and market trends.
"""

import pandas as pd
import numpy as np
import logging
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import re
from datetime import datetime, timedelta
import requests
from ..utils.database import DatabaseConnector

# Initialize logger
logger = logging.getLogger('marketmind.analysis.sentiment')

class SentimentAnalyzer:
    """
    Analyzes sentiment of financial news and social media content.
    """
    
    def __init__(self, config=None):
        """
        Initialize the SentimentAnalyzer with configuration.
        
        Args:
            config (dict): Configuration dictionary containing API keys and settings.
        """
        from .. import load_config
        self.config = config or load_config()
        self.db_connector = DatabaseConnector(self.config)
        
        # Download NLTK data if needed
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            logger.info("Downloading NLTK vader lexicon for sentiment analysis")
            nltk.download('vader_lexicon', quiet=True)
        
        # Initialize VADER sentiment analyzer
        self.vader = SentimentIntensityAnalyzer()
        
        # Load custom financial sentiment words if configured
        self.financial_words = self._load_financial_words()
        
        # Add custom financial words to VADER lexicon
        if self.financial_words:
            self.vader.lexicon.update(self.financial_words)
    
    def _load_financial_words(self):
        """
        Load custom financial sentiment words from configuration.
        
        Returns:
            dict: Dictionary of words and their sentiment scores
        """
        financial_words = {}
        
        try:
            fin_words_config = self.config.get('sentiment_analysis', {}).get('financial_words', {})
            
            # Process positive words
            for word, score in fin_words_config.get('positive', {}).items():
                financial_words[word] = float(score)
            
            # Process negative words
            for word, score in fin_words_config.get('negative', {}).items():
                financial_words[word] = float(score)
            
            logger.info(f"Loaded {len(financial_words)} custom financial sentiment words")
            
        except Exception as e:
            logger.error(f"Error loading financial sentiment words: {str(e)}")
        
        return financial_words
    
    def analyze_text(self, text):
        """
        Analyze sentiment of a single text item.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Dictionary containing sentiment scores
        """
        if not text or not isinstance(text, str):
            return {
                'compound': 0,
                'positive': 0,
                'negative': 0,
                'neutral': 0,
                'textblob_polarity': 0,
                'textblob_subjectivity': 0
            }
        
        # Clean text
        cleaned_text = self._clean_text(text)
        
        # Get VADER sentiment scores
        vader_scores = self.vader.polarity_scores(cleaned_text)
        
        # Get TextBlob sentiment
        blob = TextBlob(cleaned_text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        # Combine scores
        result = {
            'compound': vader_scores['compound'],
            'positive': vader_scores['pos'],
            'negative': vader_scores['neg'],
            'neutral': vader_scores['neu'],
            'textblob_polarity': textblob_polarity,
            'textblob_subjectivity': textblob_subjectivity
        }
        
        return result
    
    def _clean_text(self, text):
        """
        Clean text for sentiment analysis.
        
        Args:
            text (str): Text to clean
            
        Returns:
            str: Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        
        return text
    
    def analyze_news_data(self, news_df):
        """
        Analyze sentiment of news data.
        
        Args:
            news_df (pandas.DataFrame): DataFrame containing news data with 'title' and 'summary' columns
            
        Returns:
            pandas.DataFrame: DataFrame with added sentiment columns
        """
        if news_df is None or news_df.empty:
            logger.warning("No news data provided for sentiment analysis")
            return pd.DataFrame()
        
        result_df = news_df.copy()
        
        # Process titles
        if 'title' in result_df.columns:
            logger.info("Analyzing sentiment of news titles")
            title_sentiments = []
            
            for title in result_df['title']:
                title_sentiments.append(self.analyze_text(title))
            
            title_sentiment_df = pd.DataFrame(title_sentiments)
            
            # Prefix columns with 'title_'
            title_sentiment_df = title_sentiment_df.add_prefix('title_')
            
            # Reset index to ensure proper concatenation
            result_df.reset_index(drop=True, inplace=True)
            title_sentiment_df.reset_index(drop=True, inplace=True)
            
            # Add title sentiment scores to result dataframe
            result_df = pd.concat([result_df, title_sentiment_df], axis=1)
        
        # Process summaries
        if 'summary' in result_df.columns:
            logger.info("Analyzing sentiment of news summaries")
            summary_sentiments = []
            
            for summary in result_df['summary']:
                summary_sentiments.append(self.analyze_text(summary))
            
            summary_sentiment_df = pd.DataFrame(summary_sentiments)
            
            # Prefix columns with 'summary_'
            summary_sentiment_df = summary_sentiment_df.add_prefix('summary_')
            
            # Reset index to ensure proper concatenation
            result_df.reset_index(drop=True, inplace=True)
            summary_sentiment_df.reset_index(drop=True, inplace=True)
            
            # Add summary sentiment scores to result dataframe
            result_df = pd.concat([result_df, summary_sentiment_df], axis=1)
        
        # Calculate combined sentiment
        if 'title_compound' in result_df.columns and 'summary_compound' in result_df.columns:
            # Weight title more than summary (adjustable)
            title_weight = self.config.get('sentiment_analysis', {}).get('title_weight', 0.7)
            summary_weight = 1 - title_weight
            
            result_df['combined_compound'] = (
                title_weight * result_df['title_compound'] + 
                summary_weight * result_df['summary_compound']
            )
        
        logger.info(f"Completed sentiment analysis for {len(result_df)} news items")
        return result_df
    
    def analyze_social_data(self, social_df):
        """
        Analyze sentiment of social media data.
        
        Args:
            social_df (pandas.DataFrame): DataFrame containing social media data with 'text' column
            
        Returns:
            pandas.DataFrame: DataFrame with added sentiment columns
        """
        if social_df is None or social_df.empty:
            logger.warning("No social media data provided for sentiment analysis")
            return pd.DataFrame()
        
        result_df = social_df.copy()
        
        # Process social media posts
        if 'text' in result_df.columns:
            logger.info("Analyzing sentiment of social media posts")
            text_sentiments = []
            
            for text in result_df['text']:
                text_sentiments.append(self.analyze_text(text))
            
            text_sentiment_df = pd.DataFrame(text_sentiments)
            
            # Prefix columns with 'text_'
            text_sentiment_df = text_sentiment_df.add_prefix('text_')
            
            # Reset index to ensure proper concatenation
            result_df.reset_index(drop=True, inplace=True)
            text_sentiment_df.reset_index(drop=True, inplace=True)
            
            # Add text sentiment scores to result dataframe
            result_df = pd.concat([result_df, text_sentiment_df], axis=1)
        
        logger.info(f"Completed sentiment analysis for {len(result_df)} social media posts")
        return result_df
    
    def aggregate_sentiment_by_date(self, df, sentiment_column='compound', date_column='date'):
        """
        Aggregate sentiment scores by date.
        
        Args:
            df (pandas.DataFrame): DataFrame containing sentiment data
            sentiment_column (str): Column name of sentiment scores to aggregate
            date_column (str): Column name of date
            
        Returns:
            pandas.DataFrame: DataFrame with aggregated sentiment by date
        """
        if df is None or df.empty:
            return pd.DataFrame()
        
        if sentiment_column not in df.columns:
            logger.warning(f"Sentiment column '{sentiment_column}' not found in DataFrame")
            return pd.DataFrame()
        
        if date_column not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            logger.warning(f"Date column '{date_column}' not found in DataFrame")
            return pd.DataFrame()
        
        # Set date as index if it's not already
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df[date_column] = pd.to_datetime(df[date_column])
            df.set_index(date_column, inplace=True)
        
        # Aggregate by date
        result = df.resample('D')[sentiment_column].agg([
            ('mean', np.mean),
            ('median', np.median),
            ('count', 'count'),
            ('std', np.std)
        ])
        
        # Fill NaN values for std when count is 1
        result['std'] = result['std'].fillna(0)
        
        return result
    
    def aggregate_sentiment_by_symbol(self, df, sentiment_column='compound', symbol_column='symbol'):
        """
        Aggregate sentiment scores by stock symbol.
        
        Args:
            df (pandas.DataFrame): DataFrame containing sentiment data
            sentiment_column (str): Column name of sentiment scores to aggregate
            symbol_column (str): Column name of stock symbol
            
        Returns:
            pandas.DataFrame: DataFrame with aggregated sentiment by symbol
        """
        if df is None or df.empty:
            return pd.DataFrame()
        
        if sentiment_column not in df.columns:
            logger.warning(f"Sentiment column '{sentiment_column}' not found in DataFrame")
            return pd.DataFrame()
        
        if symbol_column not in df.columns:
            logger.warning(f"Symbol column '{symbol_column}' not found in DataFrame")
            return pd.DataFrame()
        
        # Aggregate by symbol
        result = df.groupby(symbol_column)[sentiment_column].agg([
            ('mean', np.mean),
            ('median', np.median),
            ('count', 'count'),
            ('std', np.std)
        ])
        
        # Fill NaN values for std when count is 1
        result['std'] = result['std'].fillna(0)
        
        return result
    
    def save_sentiment_data(self, df, data_type):
        """
        Save sentiment analysis results to database.
        
        Args:
            df (pandas.DataFrame): DataFrame containing sentiment data
            data_type (str): Type of data ('news' or 'social')
            
        Returns:
            bool: True if successful, False otherwise
        """
        if df is None or df.empty:
            logger.warning(f"No {data_type} sentiment data to save")
            return False
        
        table_name = f"{data_type}_sentiment"
        
        try:
            self.db_connector.save_dataframe(df, table_name)
            logger.info(f"Saved {len(df)} {data_type} sentiment records to database")
            return True
        except Exception as e:
            logger.error(f"Error saving {data_type} sentiment data to database: {str(e)}")
            return False
    
    def load_sentiment_data(self, data_type, start_date=None, end_date=None, symbols=None):
        """
        Load sentiment data from database.
        
        Args:
            data_type (str): Type of data ('news' or 'social')
            start_date (str, optional): Start date in format 'YYYY-MM-DD'
            end_date (str, optional): End date in format 'YYYY-MM-DD'
            symbols (list, optional): List of stock symbols to filter by
            
        Returns:
            pandas.DataFrame: DataFrame containing sentiment data
        """
        table_name = f"{data_type}_sentiment"
        
        query = f"SELECT * FROM {table_name}"
        conditions = []
        
        if start_date:
            conditions.append(f"date >= '{start_date}'")
        if end_date:
            conditions.append(f"date <= '{end_date}'")
        if symbols and isinstance(symbols, list) and len(symbols) > 0:
            symbols_str = "', '".join(symbols)
            conditions.append(f"symbol IN ('{symbols_str}')")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        try:
            df = self.db_connector.run_query(query)
            return df
        except Exception as e:
            logger.error(f"Error loading {data_type} sentiment data from database: {str(e)}")
            return None

    def merge_with_stock_data(self, stock_df, data_types=None, aggregate=True):
        """
        Merge sentiment data with stock data.
        
        Args:
            stock_df (pandas.DataFrame): Stock data DataFrame with datetime index
            data_types (list, optional): List of data types to include ('news', 'social'), defaults to both
            aggregate (bool): Whether to aggregate sentiment by date
            
        Returns:
            pandas.DataFrame: DataFrame with stock data and sentiment
        """
        if not isinstance(stock_df.index, pd.DatetimeIndex):
            logger.error("Stock data must have a DatetimeIndex")
            return stock_df
        
        if data_types is None:
            data_types = ['news', 'social']
        
        result_df = stock_df.copy()
        
        for data_type in data_types:
            try:
                # Get the earliest and latest dates from stock data
                start_date = stock_df.index.min().strftime('%Y-%m-%d')
                end_date = stock_df.index.max().strftime('%Y-%m-%d')
                
                # Get symbols if available
                symbols = None
                if 'symbol' in stock_df.columns:
                    symbols = stock_df['symbol'].unique().tolist()
                
                # Load sentiment data
                sentiment_df = self.load_sentiment_data(data_type, start_date, end_date, symbols)
                
                if sentiment_df is None or sentiment_df.empty:
                    logger.warning(f"No {data_type} sentiment data available for the specified period")
                    continue
                
                # Ensure sentiment data has a datetime index
                if not isinstance(sentiment_df.index, pd.DatetimeIndex):
                    if 'date' in sentiment_df.columns:
                        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
                        sentiment_df.set_index('date', inplace=True)
                    else:
                        logger.warning(f"Cannot set datetime index for {data_type} sentiment data")
                        continue
                
                # Aggregate sentiment by date if requested
                if aggregate:
                    compound_col = f"combined_compound" if "combined_compound" in sentiment_df.columns else "text_compound"
                    
                    if compound_col not in sentiment_df.columns:
                        compound_options = [col for col in sentiment_df.columns if 'compound' in col]
                        if compound_options:
                            compound_col = compound_options[0]
                        else:
                            logger.warning(f"No compound sentiment column found in {data_type} data")
                            continue
                    
                    agg_sentiment = self.aggregate_sentiment_by_date(sentiment_df, compound_col)
                    
                    # Rename columns for clarity
                    agg_sentiment.columns = [f"{data_type}_{col}" for col in agg_sentiment.columns]
                    
                    # Merge with stock data
                    result_df = pd.merge(result_df, agg_sentiment,
                                         left_index=True, right_index=True,
                                         how='left')
                else:
                    # Use all sentiment data
                    # Filter relevant columns
                    cols_to_include = [col for col in sentiment_df.columns if any(
                        x in col for x in ['compound', 'positive', 'negative', 'polarity']
                    )]
                    
                    sentiment_subset = sentiment_df[cols_to_include]
                    
                    # Rename columns for clarity
                    sentiment_subset.columns = [f"{data_type}_{col}" for col in sentiment_subset.columns]
                    
                    # Merge with stock data
                    result_df = pd.merge(result_df, sentiment_subset,
                                         left_index=True, right_index=True,
                                         how='left')
                
                # Fill missing values with forward fill, then backward fill
                for col in [c for c in result_df.columns if data_type in c]:
                    result_df[col] = result_df[col].fillna(method='ffill')
                    result_df[col] = result_df[col].fillna(method='bfill')
                
                logger.info(f"Merged {data_type} sentiment data")
                
            except Exception as e:
                logger.error(f"Error merging {data_type} sentiment data: {str(e)}")
        
        return result_df 