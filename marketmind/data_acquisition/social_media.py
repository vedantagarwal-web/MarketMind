"""
Social Media Scraper

This module handles scraping stock-related content from social media platforms like Twitter/X, Reddit, etc.
"""

import pandas as pd
import numpy as np
import logging
import requests
import json
import time
from datetime import datetime, timedelta
import re
from bs4 import BeautifulSoup
import praw
from ..utils.database import DatabaseConnector

# Initialize logger
logger = logging.getLogger('marketmind.data_acquisition.social_media')

class SocialMediaScraper:
    """
    Scrapes stock-related content from various social media platforms.
    """
    
    def __init__(self, config=None):
        """
        Initialize the SocialMediaScraper with configuration.
        
        Args:
            config (dict): Configuration dictionary containing API keys and settings.
        """
        from .. import load_config
        self.config = config or load_config()
        self.db_connector = DatabaseConnector(self.config)
        
        # Configure Reddit client
        reddit_config = self.config.get('api', {}).get('reddit', {})
        self.reddit_enabled = reddit_config.get('enabled', False)
        
        if self.reddit_enabled:
            try:
                self.reddit = praw.Reddit(
                    client_id=reddit_config.get('client_id'),
                    client_secret=reddit_config.get('client_secret'),
                    user_agent=reddit_config.get('user_agent', 'MarketMind Social Scraper')
                )
                logger.info("Reddit API client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Reddit API client: {str(e)}")
                self.reddit_enabled = False
        
        # Configure Twitter/X client
        twitter_config = self.config.get('api', {}).get('twitter', {})
        self.twitter_enabled = twitter_config.get('enabled', False)
        self.twitter_bearer_token = twitter_config.get('bearer_token')
        
        if self.twitter_enabled and not self.twitter_bearer_token:
            logger.warning("Twitter API enabled but no bearer token provided")
            self.twitter_enabled = False
        
        # Get stock watchlist
        self.stock_symbols = self.config.get('data', {}).get('stocks', {}).get('watchlist', [])
        self.subreddits = self.config.get('data', {}).get('social_media', {}).get('subreddits', ['wallstreetbets', 'stocks', 'investing'])
        self.max_posts_per_sub = self.config.get('data', {}).get('social_media', {}).get('max_posts_per_subreddit', 100)
        self.max_comments_per_post = self.config.get('data', {}).get('social_media', {}).get('max_comments_per_post', 50)
    
    def scrape_reddit(self, subreddits=None, time_filter='day', limit=None):
        """
        Scrape posts from specified subreddits.
        
        Args:
            subreddits (list): List of subreddit names to scrape
            time_filter (str): Time filter for posts ('day', 'week', 'month', 'year', 'all')
            limit (int): Maximum number of posts to retrieve per subreddit
            
        Returns:
            pandas.DataFrame: DataFrame containing scraped posts
        """
        if not self.reddit_enabled:
            logger.warning("Reddit API is not enabled or configured")
            return pd.DataFrame()
        
        if not subreddits:
            subreddits = self.subreddits
        
        if not limit:
            limit = self.max_posts_per_sub
        
        all_posts = []
        
        for subreddit_name in subreddits:
            try:
                logger.info(f"Scraping up to {limit} posts from r/{subreddit_name}")
                
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Get top posts from the subreddit
                for post in subreddit.top(time_filter=time_filter, limit=limit):
                    # Extract post data
                    post_data = {
                        'id': post.id,
                        'title': post.title,
                        'text': post.selftext,
                        'score': post.score,
                        'upvote_ratio': post.upvote_ratio,
                        'num_comments': post.num_comments,
                        'created_utc': datetime.utcfromtimestamp(post.created_utc),
                        'subreddit': subreddit_name,
                        'permalink': post.permalink,
                        'url': post.url,
                        'author': str(post.author),
                        'is_self': post.is_self
                    }
                    
                    # Add post to list
                    all_posts.append(post_data)
                
                # Add some sleep to avoid hitting rate limits
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error scraping r/{subreddit_name}: {str(e)}")
        
        # Convert to DataFrame
        if all_posts:
            df = pd.DataFrame(all_posts)
            logger.info(f"Scraped {len(df)} Reddit posts from {len(subreddits)} subreddits")
            return df
        else:
            logger.warning("No Reddit posts were scraped")
            return pd.DataFrame()
    
    def scrape_reddit_comments(self, posts_df, limit=None):
        """
        Scrape comments from Reddit posts.
        
        Args:
            posts_df (pandas.DataFrame): DataFrame containing Reddit posts with 'id' column
            limit (int): Maximum number of comments to retrieve per post
            
        Returns:
            pandas.DataFrame: DataFrame containing scraped comments
        """
        if not self.reddit_enabled:
            logger.warning("Reddit API is not enabled or configured")
            return pd.DataFrame()
        
        if posts_df.empty or 'id' not in posts_df.columns:
            logger.warning("No posts provided or missing 'id' column")
            return pd.DataFrame()
        
        if not limit:
            limit = self.max_comments_per_post
        
        all_comments = []
        
        for _, post_row in posts_df.iterrows():
            try:
                post_id = post_row['id']
                subreddit = post_row.get('subreddit', '')
                
                logger.info(f"Scraping up to {limit} comments from post {post_id} in r/{subreddit}")
                
                # Get submission by ID
                submission = self.reddit.submission(id=post_id)
                
                # Replace MoreComments objects with actual comments
                submission.comments.replace_more(limit=0)
                
                # Get top comments
                comment_count = 0
                for comment in submission.comments.list()[:limit]:
                    # Extract comment data
                    comment_data = {
                        'id': comment.id,
                        'post_id': post_id,
                        'subreddit': subreddit,
                        'text': comment.body,
                        'score': comment.score,
                        'created_utc': datetime.utcfromtimestamp(comment.created_utc),
                        'author': str(comment.author),
                        'is_submitter': comment.is_submitter,
                        'parent_id': comment.parent_id
                    }
                    
                    # Add comment to list
                    all_comments.append(comment_data)
                    comment_count += 1
                
                logger.info(f"Scraped {comment_count} comments from post {post_id}")
                
                # Add some sleep to avoid hitting rate limits
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error scraping comments for post {post_row.get('id', 'unknown')}: {str(e)}")
        
        # Convert to DataFrame
        if all_comments:
            df = pd.DataFrame(all_comments)
            logger.info(f"Scraped {len(df)} Reddit comments from {len(posts_df)} posts")
            return df
        else:
            logger.warning("No Reddit comments were scraped")
            return pd.DataFrame()
    
    def scrape_twitter(self, query=None, max_results=100, days_back=7):
        """
        Scrape tweets using the Twitter/X API.
        
        Args:
            query (str): Search query for tweets
            max_results (int): Maximum number of tweets to retrieve
            days_back (int): Number of days back to search for tweets
            
        Returns:
            pandas.DataFrame: DataFrame containing scraped tweets
        """
        if not self.twitter_enabled:
            logger.warning("Twitter API is not enabled or configured")
            return pd.DataFrame()
        
        if not query:
            if not self.stock_symbols:
                logger.warning("No stock symbols provided for Twitter search")
                return pd.DataFrame()
            
            # Create query from stock symbols
            stock_query_parts = []
            for symbol in self.stock_symbols[:5]:  # Limit to 5 symbols to avoid query complexity
                stock_query_parts.append(f"${symbol}")
            
            query = " OR ".join(stock_query_parts)
        
        # Set up search params
        start_time = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        search_url = "https://api.twitter.com/2/tweets/search/recent"
        
        # Set query parameters
        query_params = {
            'query': query,
            'start_time': start_time,
            'max_results': min(100, max_results),  # API limit is 100 per request
            'tweet.fields': 'created_at,public_metrics,author_id',
            'expansions': 'author_id',
            'user.fields': 'username,name,public_metrics'
        }
        
        headers = {
            'Authorization': f"Bearer {self.twitter_bearer_token}",
            'Content-Type': 'application/json'
        }
        
        all_tweets = []
        total_count = 0
        
        # Paginate through results
        next_token = None
        
        while total_count < max_results:
            if next_token:
                query_params['next_token'] = next_token
            
            try:
                logger.info(f"Requesting Twitter API with query: {query}")
                
                response = requests.get(search_url, headers=headers, params=query_params)
                
                if response.status_code != 200:
                    logger.error(f"Twitter API request failed with status {response.status_code}: {response.text}")
                    break
                
                response_data = response.json()
                
                # Process tweets
                if 'data' in response_data:
                    # Create user lookup
                    users = {}
                    if 'includes' in response_data and 'users' in response_data['includes']:
                        for user in response_data['includes']['users']:
                            users[user['id']] = user
                    
                    # Process tweets
                    for tweet in response_data['data']:
                        author_id = tweet.get('author_id')
                        user = users.get(author_id, {})
                        
                        tweet_data = {
                            'id': tweet['id'],
                            'text': tweet['text'],
                            'created_at': datetime.strptime(tweet['created_at'], "%Y-%m-%dT%H:%M:%S.%fZ"),
                            'author_id': author_id,
                            'username': user.get('username', ''),
                            'name': user.get('name', ''),
                            'retweet_count': tweet.get('public_metrics', {}).get('retweet_count', 0),
                            'reply_count': tweet.get('public_metrics', {}).get('reply_count', 0),
                            'like_count': tweet.get('public_metrics', {}).get('like_count', 0),
                            'quote_count': tweet.get('public_metrics', {}).get('quote_count', 0),
                            'followers_count': user.get('public_metrics', {}).get('followers_count', 0)
                        }
                        
                        all_tweets.append(tweet_data)
                        total_count += 1
                
                # Check if there are more results
                if 'meta' in response_data and 'next_token' in response_data['meta']:
                    next_token = response_data['meta']['next_token']
                else:
                    break
                
                # Add some sleep to avoid hitting rate limits
                time.sleep(1)
                
                # If we've reached the max results, break
                if total_count >= max_results:
                    break
                
            except Exception as e:
                logger.error(f"Error scraping Twitter: {str(e)}")
                break
        
        # Convert to DataFrame
        if all_tweets:
            df = pd.DataFrame(all_tweets)
            logger.info(f"Scraped {len(df)} tweets with query: {query}")
            return df
        else:
            logger.warning("No tweets were scraped")
            return pd.DataFrame()
    
    def filter_by_stocks(self, df, text_column='text', title_column=None):
        """
        Filter social media data to include only posts/tweets mentioning watched stocks.
        
        Args:
            df (pandas.DataFrame): DataFrame containing social media data
            text_column (str): Column name containing the main text
            title_column (str, optional): Column name containing the title (if applicable)
            
        Returns:
            pandas.DataFrame: Filtered DataFrame
        """
        if df.empty or not self.stock_symbols:
            return df
        
        logger.info("Filtering social media data by stock symbols")
        
        # Create a copy of the DataFrame
        result_df = df.copy()
        
        # Add a new column for the mentioned symbols
        result_df['mentioned_symbols'] = [[] for _ in range(len(result_df))]
        
        # Create patterns for stock symbols
        # Match $SYMBOL format and "SYMBOL" keyword
        symbol_patterns = {}
        for symbol in self.stock_symbols:
            # Match both $SYMBOL and word boundaries for SYMBOL
            symbol_patterns[symbol] = re.compile(r'(\$' + re.escape(symbol) + r'\b|\b' + re.escape(symbol) + r'\b)', re.IGNORECASE)
        
        # Function to find mentioned symbols in text
        def find_symbols(text):
            if not isinstance(text, str):
                return []
            
            found_symbols = []
            for symbol, pattern in symbol_patterns.items():
                if pattern.search(text):
                    found_symbols.append(symbol)
            
            return found_symbols
        
        # Check for symbols in text column
        for i, row in result_df.iterrows():
            text = row.get(text_column, '')
            symbols = find_symbols(text)
            
            # Also check title if provided
            if title_column and title_column in row:
                title = row.get(title_column, '')
                symbols.extend(find_symbols(title))
            
            # Remove duplicates
            symbols = list(set(symbols))
            
            result_df.at[i, 'mentioned_symbols'] = symbols
        
        # Filter to include only rows that mention at least one symbol
        filtered_df = result_df[result_df['mentioned_symbols'].map(len) > 0]
        
        logger.info(f"Found {len(filtered_df)} items mentioning watched stocks out of {len(df)} total")
        
        return filtered_df
    
    def extract_sentiment_keywords(self, df, text_column='text'):
        """
        Extract sentiment-related keywords from social media text.
        
        Args:
            df (pandas.DataFrame): DataFrame containing social media data
            text_column (str): Column name containing the text
            
        Returns:
            pandas.DataFrame: DataFrame with added keyword columns
        """
        if df.empty:
            return df
        
        logger.info("Extracting sentiment keywords from social media data")
        
        # Create a copy of the DataFrame
        result_df = df.copy()
        
        # Load sentiment keywords from config
        pos_keywords = self.config.get('sentiment_analysis', {}).get('keywords', {}).get('positive', [])
        neg_keywords = self.config.get('sentiment_analysis', {}).get('keywords', {}).get('negative', [])
        
        # Initialize keyword columns
        result_df['positive_keywords'] = [[] for _ in range(len(result_df))]
        result_df['negative_keywords'] = [[] for _ in range(len(result_df))]
        
        # Compile regex patterns for efficiency
        pos_patterns = [re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE) for kw in pos_keywords]
        neg_patterns = [re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE) for kw in neg_keywords]
        
        # Process each row
        for i, row in result_df.iterrows():
            text = row.get(text_column, '')
            
            if not isinstance(text, str):
                continue
            
            # Find positive keywords
            pos_found = []
            for pattern, kw in zip(pos_patterns, pos_keywords):
                if pattern.search(text):
                    pos_found.append(kw)
            
            # Find negative keywords
            neg_found = []
            for pattern, kw in zip(neg_patterns, neg_keywords):
                if pattern.search(text):
                    neg_found.append(kw)
            
            # Update dataframe
            result_df.at[i, 'positive_keywords'] = pos_found
            result_df.at[i, 'negative_keywords'] = neg_found
        
        # Add keyword count columns
        result_df['positive_keyword_count'] = result_df['positive_keywords'].map(len)
        result_df['negative_keyword_count'] = result_df['negative_keywords'].map(len)
        
        # Add simple sentiment score (difference between positive and negative keywords)
        result_df['keyword_sentiment_score'] = result_df['positive_keyword_count'] - result_df['negative_keyword_count']
        
        return result_df
    
    def save_to_database(self, df, source):
        """
        Save scraped social media data to database.
        
        Args:
            df (pandas.DataFrame): DataFrame containing social media data
            source (str): Source of the data ('reddit', 'twitter', etc.)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if df.empty:
            logger.warning(f"No {source} data to save")
            return False
        
        table_name = f"social_{source}"
        
        try:
            # Convert lists to strings for database storage
            df_to_save = df.copy()
            for col in df_to_save.columns:
                if isinstance(df_to_save[col].iloc[0], list):
                    df_to_save[col] = df_to_save[col].apply(lambda x: ','.join(x) if x else '')
            
            self.db_connector.save_dataframe(df_to_save, table_name)
            logger.info(f"Saved {len(df)} {source} records to database")
            return True
        except Exception as e:
            logger.error(f"Error saving {source} data to database: {str(e)}")
            return False
    
    def load_from_database(self, source, start_date=None, end_date=None, symbols=None):
        """
        Load social media data from database.
        
        Args:
            source (str): Source of the data ('reddit', 'twitter', etc.)
            start_date (str, optional): Start date in format 'YYYY-MM-DD'
            end_date (str, optional): End date in format 'YYYY-MM-DD'
            symbols (list, optional): List of stock symbols to filter by
            
        Returns:
            pandas.DataFrame: DataFrame containing social media data
        """
        table_name = f"social_{source}"
        
        query = f"SELECT * FROM {table_name}"
        conditions = []
        
        # Add date conditions based on the appropriate date column for each source
        date_column = 'created_utc' if source == 'reddit' else 'created_at'
        
        if start_date:
            conditions.append(f"{date_column} >= '{start_date}'")
        if end_date:
            conditions.append(f"{date_column} <= '{end_date}'")
        
        # Add symbol condition if provided
        if symbols and isinstance(symbols, list) and len(symbols) > 0:
            # For symbols, we need to check if the mentioned_symbols column contains any of the requested symbols
            # This is more complex in SQL, so we'll filter after loading the data
            pass
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        try:
            df = self.db_connector.run_query(query)
            
            # Convert string representation of lists back to actual lists
            for col in ['mentioned_symbols', 'positive_keywords', 'negative_keywords']:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: x.split(',') if isinstance(x, str) and x else [])
            
            # Filter by symbols if provided
            if symbols and isinstance(symbols, list) and len(symbols) > 0:
                df = df[df['mentioned_symbols'].apply(lambda x: any(s in x for s in symbols))]
            
            return df
        except Exception as e:
            logger.error(f"Error loading {source} data from database: {str(e)}")
            return None
    
    def scrape_all(self, save=True):
        """
        Scrape data from all configured social media sources.
        
        Args:
            save (bool): Whether to save the data to database
            
        Returns:
            dict: Dictionary of DataFrames keyed by source
        """
        results = {}
        
        # Scrape Reddit
        if self.reddit_enabled:
            try:
                # Scrape posts
                reddit_posts = self.scrape_reddit()
                
                if not reddit_posts.empty:
                    # Filter by stock symbols
                    filtered_posts = self.filter_by_stocks(reddit_posts, 'text', 'title')
                    
                    # Extract sentiment keywords
                    processed_posts = self.extract_sentiment_keywords(filtered_posts, 'text')
                    
                    # Scrape comments for filtered posts
                    comments = self.scrape_reddit_comments(filtered_posts)
                    
                    if not comments.empty:
                        # Filter and process comments
                        filtered_comments = self.filter_by_stocks(comments, 'text')
                        processed_comments = self.extract_sentiment_keywords(filtered_comments, 'text')
                        
                        # Save to database if requested
                        if save:
                            self.save_to_database(processed_comments, 'reddit_comments')
                        
                        results['reddit_comments'] = processed_comments
                    
                    # Save posts to database if requested
                    if save:
                        self.save_to_database(processed_posts, 'reddit_posts')
                    
                    results['reddit_posts'] = processed_posts
            
            except Exception as e:
                logger.error(f"Error in Reddit scraping workflow: {str(e)}")
        
        # Scrape Twitter
        if self.twitter_enabled:
            try:
                # Scrape tweets
                tweets = self.scrape_twitter()
                
                if not tweets.empty:
                    # Filter by stock symbols
                    filtered_tweets = self.filter_by_stocks(tweets, 'text')
                    
                    # Extract sentiment keywords
                    processed_tweets = self.extract_sentiment_keywords(filtered_tweets, 'text')
                    
                    # Save to database if requested
                    if save:
                        self.save_to_database(processed_tweets, 'twitter')
                    
                    results['twitter'] = processed_tweets
            
            except Exception as e:
                logger.error(f"Error in Twitter scraping workflow: {str(e)}")
        
        return results 