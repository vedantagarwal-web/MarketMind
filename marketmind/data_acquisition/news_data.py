"""
News Data Fetcher

This module handles fetching news articles for sentiment analysis from various sources.
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
from bs4 import BeautifulSoup
from newspaper import Article
import nltk
from ..utils.database import DatabaseConnector

logger = logging.getLogger('marketmind.data_acquisition.news_data')

class NewsDataFetcher:
    """
    Fetches news articles from various sources for sentiment analysis.
    """
    
    def __init__(self, config=None):
        """
        Initialize the NewsDataFetcher with configuration.
        
        Args:
            config (dict): Configuration dictionary containing API keys and settings.
        """
        from .. import load_config
        self.config = config or load_config()
        self.api_key = self.config.get('api', {}).get('news_api', {}).get('key')
        self.base_url = self.config.get('api', {}).get('news_api', {}).get('base_url', 'https://newsapi.org/v2/')
        self.max_articles = self.config.get('data', {}).get('news', {}).get('max_articles_per_day', 20)
        self.sources = self.config.get('data', {}).get('news', {}).get('sources', [])
        self.db_connector = DatabaseConnector(self.config)
        
        # Download NLTK data if not already present
        try:
            nltk.data.find('punkt')
        except LookupError:
            nltk.download('punkt')
        
        if not self.api_key or self.api_key == "YOUR_NEWS_API_KEY":
            logger.warning("News API key not set. Please set your API key in config.yaml")
    
    def fetch_news_by_keyword(self, keywords, start_date=None, end_date=None, save=True):
        """
        Fetch news articles by keyword search.
        
        Args:
            keywords (str or list): Keywords to search for (e.g., 'Apple', ['Tesla', 'EV'])
            start_date (str, optional): Start date in format 'YYYY-MM-DD'
            end_date (str, optional): End date in format 'YYYY-MM-DD'
            save (bool): Whether to save the data to database
            
        Returns:
            pandas.DataFrame: DataFrame containing news articles
        """
        if not self.api_key:
            logger.error("Cannot fetch news: API key not set")
            return None
        
        # Format keywords
        if isinstance(keywords, list):
            query = ' OR '.join(keywords)
        else:
            query = keywords
        
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            # Default to 7 days before end_date
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            start_dt = end_dt - timedelta(days=7)
            start_date = start_dt.strftime('%Y-%m-%d')
        
        logger.info(f"Fetching news articles for keywords: {keywords} from {start_date} to {end_date}")
        
        params = {
            'q': query,
            'from': start_date,
            'to': end_date,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': self.max_articles,
            'apiKey': self.api_key
        }
        
        # Add sources if configured
        if self.sources:
            sources_str = ','.join(self.sources)
            params['sources'] = sources_str
        
        try:
            endpoint = self.base_url + 'everything'
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Check for error messages
            if 'status' in data and data['status'] != 'ok':
                logger.error(f"API returned error: {data.get('message', 'Unknown error')}")
                return None
            
            articles = data.get('articles', [])
            
            if not articles:
                logger.warning(f"No news articles found for keywords: {keywords}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(articles)
            
            # Extract source name from source dictionary
            df['source'] = df['source'].apply(lambda x: x.get('name', 'Unknown'))
            
            # Add search keywords column
            df['keywords'] = str(keywords) if isinstance(keywords, list) else keywords
            
            # Add fetch timestamp
            df['fetch_timestamp'] = datetime.now().isoformat()
            
            # Extract full text for select articles (limit due to potential rate limits)
            self._extract_full_text(df)
            
            # Save to database if requested
            if save:
                keyword_str = '_'.join(keywords) if isinstance(keywords, list) else keywords
                table_name = f"news_{keyword_str}"
                self._save_to_database(df, table_name)
            
            logger.info(f"Successfully fetched {len(df)} news articles for keywords: {keywords}")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching news articles: {str(e)}")
            return None
        
        except ValueError as e:
            logger.error(f"Error parsing JSON response: {str(e)}")
            return None
    
    def fetch_company_news(self, company_symbols, start_date=None, end_date=None, save=True):
        """
        Fetch news for a list of companies.
        
        Args:
            company_symbols (list): List of company symbols
            start_date (str, optional): Start date in format 'YYYY-MM-DD'
            end_date (str, optional): End date in format 'YYYY-MM-DD'
            save (bool): Whether to save the data to database
            
        Returns:
            dict: Dictionary of DataFrames keyed by company symbol
        """
        results = {}
        
        # Map of company symbols to search terms for better news coverage
        company_terms = {
            'AAPL': ['Apple', 'iPhone', 'Tim Cook', 'MacBook', 'iOS'],
            'MSFT': ['Microsoft', 'Windows', 'Satya Nadella', 'Xbox', 'Teams', 'Azure'],
            'GOOGL': ['Google', 'Alphabet', 'Android', 'Sundar Pichai', 'Chrome'],
            'AMZN': ['Amazon', 'AWS', 'Jeff Bezos', 'Andy Jassy', 'Prime'],
            'META': ['Facebook', 'Meta', 'Mark Zuckerberg', 'Instagram', 'WhatsApp']
        }
        
        for symbol in company_symbols:
            # Use predefined terms if available, otherwise use the symbol
            search_terms = company_terms.get(symbol, [symbol])
            
            # Fetch news for this company
            df = self.fetch_news_by_keyword(search_terms, start_date, end_date, save)
            
            if df is not None:
                results[symbol] = df
        
        return results
    
    def _extract_full_text(self, df, max_articles=10):
        """
        Extract full text of articles using newspaper3k.
        
        Args:
            df (pandas.DataFrame): DataFrame containing articles with 'url' column
            max_articles (int): Maximum number of articles to extract full text for
        """
        # Add full_text column
        df['full_text'] = None
        
        # Limit to prevent too many requests
        urls_to_process = df['url'].head(max_articles).tolist()
        
        for i, url in enumerate(urls_to_process):
            try:
                # Download and parse article
                article = Article(url)
                article.download()
                article.parse()
                
                # Update DataFrame
                df.loc[df['url'] == url, 'full_text'] = article.text
                
                logger.debug(f"Extracted full text from article {i+1}/{len(urls_to_process)}")
                
                # Respect website policies with a small delay
                time.sleep(1)
                
            except Exception as e:
                logger.warning(f"Error extracting text from {url}: {str(e)}")
    
    def web_scrape_financial_news(self, source, num_articles=10, save=True):
        """
        Web scrape financial news from specific websites.
        
        Args:
            source (str): Source to scrape (e.g., 'yahoo-finance', 'cnbc', 'bloomberg')
            num_articles (int): Number of articles to scrape
            save (bool): Whether to save data to database
            
        Returns:
            pandas.DataFrame: DataFrame containing scraped news
        """
        scraper_map = {
            'yahoo-finance': self._scrape_yahoo_finance,
            'cnbc': self._scrape_cnbc,
            'bloomberg': self._scrape_bloomberg
        }
        
        if source not in scraper_map:
            logger.error(f"Unsupported news source: {source}")
            return None
        
        try:
            # Call the appropriate scraper function
            articles = scraper_map[source](num_articles)
            
            if not articles:
                logger.warning(f"No articles scraped from {source}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(articles)
            
            # Add source and timestamp columns
            df['source'] = source
            df['fetch_timestamp'] = datetime.now().isoformat()
            
            # Save to database if requested
            if save:
                table_name = f"news_scraped_{source}"
                self._save_to_database(df, table_name)
            
            logger.info(f"Successfully scraped {len(df)} articles from {source}")
            return df
            
        except Exception as e:
            logger.error(f"Error scraping {source}: {str(e)}")
            return None
    
    def _scrape_yahoo_finance(self, num_articles=10):
        """
        Scrape news from Yahoo Finance.
        
        Args:
            num_articles (int): Maximum number of articles to scrape
            
        Returns:
            list: List of article dictionaries
        """
        articles = []
        url = 'https://finance.yahoo.com/news/'
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Yahoo Finance specific selectors
            article_elements = soup.select('div.Ov\(h\) article')[:num_articles]
            
            for article in article_elements:
                try:
                    # Extract title
                    title_elem = article.select_one('h3')
                    title = title_elem.text if title_elem else 'No title'
                    
                    # Extract link
                    link_elem = article.select_one('a')
                    link = 'https://finance.yahoo.com' + link_elem['href'] if link_elem and 'href' in link_elem.attrs else None
                    
                    # Extract summary (if available)
                    summary_elem = article.select_one('p')
                    summary = summary_elem.text if summary_elem else None
                    
                    # Extract publish date (Yahoo format varies)
                    date_elem = article.select_one('div.C\(#959595\)')
                    publish_date = date_elem.text if date_elem else datetime.now().isoformat()
                    
                    # Add to articles list
                    articles.append({
                        'title': title,
                        'url': link,
                        'description': summary,
                        'publishedAt': publish_date,
                        'content': None  # Will be populated later if needed
                    })
                    
                except Exception as e:
                    logger.warning(f"Error processing Yahoo Finance article: {str(e)}")
            
            return articles
            
        except Exception as e:
            logger.error(f"Error scraping Yahoo Finance: {str(e)}")
            return []
    
    def _scrape_cnbc(self, num_articles=10):
        """
        Scrape news from CNBC.
        
        Args:
            num_articles (int): Maximum number of articles to scrape
            
        Returns:
            list: List of article dictionaries
        """
        articles = []
        url = 'https://www.cnbc.com/technology/'
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # CNBC specific selectors
            article_elements = soup.select('div.Card-titleContainer')[:num_articles]
            
            for article in article_elements:
                try:
                    # Extract title
                    title_elem = article.select_one('a.Card-title')
                    title = title_elem.text.strip() if title_elem else 'No title'
                    
                    # Extract link
                    link = title_elem['href'] if title_elem and 'href' in title_elem.attrs else None
                    
                    # Extract time (if available)
                    time_elem = article.select_one('time')
                    publish_date = time_elem['datetime'] if time_elem and 'datetime' in time_elem.attrs else datetime.now().isoformat()
                    
                    # Add to articles list
                    articles.append({
                        'title': title,
                        'url': link,
                        'description': None,  # No summary readily available
                        'publishedAt': publish_date,
                        'content': None  # Will be populated later if needed
                    })
                    
                except Exception as e:
                    logger.warning(f"Error processing CNBC article: {str(e)}")
            
            return articles
            
        except Exception as e:
            logger.error(f"Error scraping CNBC: {str(e)}")
            return []
    
    def _scrape_bloomberg(self, num_articles=10):
        """
        Scrape news from Bloomberg.
        
        Args:
            num_articles (int): Maximum number of articles to scrape
            
        Returns:
            list: List of article dictionaries
        """
        articles = []
        url = 'https://www.bloomberg.com/technology'
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Bloomberg specific selectors
            article_elements = soup.select('article.story-package-module__story')[:num_articles]
            
            for article in article_elements:
                try:
                    # Extract title
                    title_elem = article.select_one('h3.story-package-module__headline')
                    title = title_elem.text.strip() if title_elem else 'No title'
                    
                    # Extract link
                    link_elem = article.select_one('a.story-package-module__headline')
                    base_url = 'https://www.bloomberg.com'
                    link = base_url + link_elem['href'] if link_elem and 'href' in link_elem.attrs else None
                    
                    # Extract summary (if available)
                    summary_elem = article.select_one('p.story-package-module__summary')
                    summary = summary_elem.text.strip() if summary_elem else None
                    
                    # Bloomberg doesn't readily display timestamps on article cards
                    publish_date = datetime.now().isoformat()
                    
                    # Add to articles list
                    articles.append({
                        'title': title,
                        'url': link,
                        'description': summary,
                        'publishedAt': publish_date,
                        'content': None  # Will be populated later if needed
                    })
                    
                except Exception as e:
                    logger.warning(f"Error processing Bloomberg article: {str(e)}")
            
            return articles
            
        except Exception as e:
            logger.error(f"Error scraping Bloomberg: {str(e)}")
            return []
    
    def _save_to_database(self, df, table_name):
        """
        Save the fetched news data to database.
        
        Args:
            df (pandas.DataFrame): DataFrame to save
            table_name (str): Table name for the data
        """
        try:
            self.db_connector.save_dataframe(df, table_name)
            logger.info(f"Saved news data to database table: {table_name}")
        except Exception as e:
            logger.error(f"Error saving news data to database: {str(e)}")
    
    def load_from_database(self, keywords=None, start_date=None, end_date=None):
        """
        Load news data from database.
        
        Args:
            keywords (str or list, optional): Keywords to filter by
            start_date (str, optional): Start date in format 'YYYY-MM-DD'
            end_date (str, optional): End date in format 'YYYY-MM-DD'
            
        Returns:
            pandas.DataFrame: DataFrame containing the news data
        """
        # Determine table name if keywords provided
        table_name = None
        if keywords:
            keyword_str = '_'.join(keywords) if isinstance(keywords, list) else keywords
            table_name = f"news_{keyword_str}"
        
        if table_name:
            # Load from specific table
            query = f"SELECT * FROM {table_name}"
            conditions = []
            
            if start_date:
                conditions.append(f"publishedAt >= '{start_date}'")
            if end_date:
                conditions.append(f"publishedAt <= '{end_date}'")
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
        else:
            # Query across all news tables
            tables = self.db_connector.get_tables(pattern='news_%')
            query_parts = []
            
            for table in tables:
                query_part = f"SELECT * FROM {table}"
                conditions = []
                
                if start_date:
                    conditions.append(f"publishedAt >= '{start_date}'")
                if end_date:
                    conditions.append(f"publishedAt <= '{end_date}'")
                
                if conditions:
                    query_part += " WHERE " + " AND ".join(conditions)
                
                query_parts.append(query_part)
            
            query = " UNION ALL ".join(query_parts) if query_parts else None
        
        if not query:
            logger.error("Could not construct a valid query for news data")
            return None
        
        try:
            df = self.db_connector.run_query(query)
            return df
        except Exception as e:
            logger.error(f"Error loading news data from database: {str(e)}")
            return None 