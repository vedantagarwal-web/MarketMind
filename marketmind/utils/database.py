"""
Database Utility Module

This module provides database connectivity and operations for MarketMind.
"""

import sqlite3
import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path
from datetime import datetime, date

# Try importing MySQL connector, but handle missing package gracefully
try:
    import mysql.connector
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False
    logging.getLogger('marketmind.utils.database').warning(
        "mysql-connector package not found. MySQL support will not be available. "
        "Install with: pip install mysql-connector-python"
    )

# Initialize logger
logger = logging.getLogger('marketmind.utils.database')

class DatabaseConnector:
    """
    Database connection and operations handler.
    Supports SQLite and MySQL connections.
    """
    
    def __init__(self, config=None):
        """
        Initialize the DatabaseConnector with configuration.
        
        Args:
            config (dict): Configuration dictionary containing database settings.
        """
        from .. import load_config
        self.config = config or load_config()
        self.connection = None
        self.db_type = self.config.get('database', {}).get('type', 'sqlite').lower()
        
        # Configure connection based on database type
        if self.db_type == 'sqlite':
            self._configure_sqlite()
        elif self.db_type == 'mysql':
            if not MYSQL_AVAILABLE:
                logger.warning("MySQL support not available. Falling back to SQLite.")
                self.db_type = 'sqlite'
                self._configure_sqlite()
            else:
                self._configure_mysql()
        else:
            logger.error(f"Unsupported database type: {self.db_type}")
            logger.info("Falling back to SQLite database")
            self.db_type = 'sqlite'
            self._configure_sqlite()
    
    def _configure_sqlite(self):
        """Configure SQLite database connection."""
        db_path = self.config.get('database', {}).get('sqlite', {}).get('path', 'marketmind.db')
        
        # Ensure the directory exists
        if not os.path.isabs(db_path):
            # Use the config's directory as base
            base_dir = self.config.get('base_dir', os.path.expanduser('~/.marketmind'))
            db_path = os.path.join(base_dir, db_path)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        logger.info(f"Configured SQLite database at: {self.db_path}")
    
    def _configure_mysql(self):
        """Configure MySQL database connection."""
        self.mysql_config = {
            'host': self.config.get('database', {}).get('mysql', {}).get('host', 'localhost'),
            'port': self.config.get('database', {}).get('mysql', {}).get('port', 3306),
            'database': self.config.get('database', {}).get('mysql', {}).get('database', 'marketmind'),
            'user': self.config.get('database', {}).get('mysql', {}).get('user', 'root'),
            'password': self.config.get('database', {}).get('mysql', {}).get('password', ''),
            'auth_plugin': self.config.get('database', {}).get('mysql', {}).get('auth_plugin', 'mysql_native_password')
        }
        
        logger.info(f"Configured MySQL database at: {self.mysql_config['host']}:{self.mysql_config['port']}/{self.mysql_config['database']}")
    
    def connect(self):
        """
        Establish a connection to the database.
        
        Returns:
            connection: Database connection object
        """
        if self.connection:
            return self.connection
        
        try:
            if self.db_type == 'sqlite':
                self.connection = sqlite3.connect(self.db_path)
                logger.info(f"Connected to SQLite database: {self.db_path}")
            
            elif self.db_type == 'mysql':
                self.connection = mysql.connector.connect(**self.mysql_config)
                logger.info(f"Connected to MySQL database: {self.mysql_config['database']}")
            
            return self.connection
        
        except Exception as e:
            logger.error(f"Error connecting to {self.db_type} database: {str(e)}")
            raise
    
    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info(f"Closed {self.db_type} database connection")
    
    def run_query(self, query, params=None, return_df=True):
        """
        Execute a SQL query and optionally return results as a DataFrame.
        
        Args:
            query (str): SQL query to execute
            params (tuple, dict, optional): Parameters for the query
            return_df (bool): Whether to return results as DataFrame
            
        Returns:
            pandas.DataFrame or int: Query results or number of affected rows
        """
        connection = self.connect()
        
        try:
            if return_df:
                if params:
                    return pd.read_sql_query(query, connection, params=params)
                else:
                    return pd.read_sql_query(query, connection)
            else:
                cursor = connection.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                connection.commit()
                affected_rows = cursor.rowcount
                cursor.close()
                
                return affected_rows
        
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            logger.debug(f"Query: {query}")
            raise
    
    def execute_many(self, query, params_list):
        """
        Execute a batch SQL operation (many executions of the same query).
        
        Args:
            query (str): SQL query to execute
            params_list (list): List of parameter tuples/dicts
            
        Returns:
            int: Number of affected rows
        """
        connection = self.connect()
        
        try:
            cursor = connection.cursor()
            cursor.executemany(query, params_list)
            connection.commit()
            affected_rows = cursor.rowcount
            cursor.close()
            
            return affected_rows
        
        except Exception as e:
            logger.error(f"Error executing batch query: {str(e)}")
            logger.debug(f"Query: {query}")
            raise
    
    def table_exists(self, table_name):
        """
        Check if a table exists in the database.
        
        Args:
            table_name (str): Name of the table to check
            
        Returns:
            bool: True if table exists, False otherwise
        """
        try:
            if self.db_type == 'sqlite':
                query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
                result = self.run_query(query, (table_name,))
                return not result.empty
            
            elif self.db_type == 'mysql':
                query = "SHOW TABLES LIKE %s"
                result = self.run_query(query, (table_name,))
                return not result.empty
            
            return False
        
        except Exception as e:
            logger.error(f"Error checking if table {table_name} exists: {str(e)}")
            return False
    
    def create_table(self, table_name, schema):
        """
        Create a table if it doesn't exist.
        
        Args:
            table_name (str): Name of the table to create
            schema (str): SQL schema definition for the table
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.table_exists(table_name):
            logger.info(f"Table {table_name} already exists")
            return True
        
        try:
            query = f"CREATE TABLE {table_name} ({schema})"
            self.run_query(query, return_df=False)
            logger.info(f"Created table {table_name}")
            return True
        
        except Exception as e:
            logger.error(f"Error creating table {table_name}: {str(e)}")
            return False
    
    def save_dataframe(self, df, table_name, if_exists='replace'):
        """
        Save a DataFrame to a database table.
        
        Args:
            df (pandas.DataFrame): DataFrame to save
            table_name (str): Name of the table
            if_exists (str): Action if the table exists ('fail', 'replace', 'append')
            
        Returns:
            bool: True if successful, False otherwise
        """
        if df is None or df.empty:
            logger.warning(f"No data to save to table {table_name}")
            return False
        
        # Make a copy of the DataFrame to avoid modifying the original
        df_copy = df.copy()
        
        # Convert datetime columns to string ISO format for compatibility
        for col in df_copy.columns:
            # Handle DatetimeIndex
            if isinstance(df_copy.index, pd.DatetimeIndex) and col == df_copy.index.name:
                df_copy = df_copy.reset_index()
            
            # Handle datetime columns
            if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].astype(str)
            
            # Handle date columns
            elif df_copy[col].dtype == 'object':
                # Check if column contains date objects
                if not df_copy[col].empty and isinstance(df_copy[col].iloc[0], (datetime, date)):
                    df_copy[col] = df_copy[col].astype(str)
        
        # Replace NaN values with None for SQL compatibility
        df_copy = df_copy.replace({np.nan: None})
        
        try:
            connection = self.connect()
            
            if self.db_type == 'sqlite':
                df_copy.to_sql(table_name, connection, if_exists=if_exists, index=False)
            
            elif self.db_type == 'mysql':
                # For MySQL, create the table and insert data manually for more control
                if if_exists == 'replace' and self.table_exists(table_name):
                    self.run_query(f"DROP TABLE {table_name}", return_df=False)
                
                # Create table if it doesn't exist
                if not self.table_exists(table_name):
                    columns = []
                    for col in df_copy.columns:
                        dtype = df_copy[col].dtype
                        
                        if pd.api.types.is_integer_dtype(dtype):
                            col_type = "INT"
                        elif pd.api.types.is_float_dtype(dtype):
                            col_type = "FLOAT"
                        elif pd.api.types.is_bool_dtype(dtype):
                            col_type = "BOOLEAN"
                        else:
                            # For datetime and other types, use TEXT for maximum compatibility
                            col_type = "TEXT"
                        
                        columns.append(f"`{col}` {col_type}")
                    
                    schema = ", ".join(columns)
                    self.create_table(table_name, schema)
                
                # Insert data
                placeholders = ", ".join(["%s"] * len(df_copy.columns))
                columns = ", ".join([f"`{col}`" for col in df_copy.columns])
                
                query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
                
                data = [tuple(row) for row in df_copy.values]
                self.execute_many(query, data)
            
            logger.info(f"Saved {len(df_copy)} rows to table {table_name}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving DataFrame to table {table_name}: {str(e)}")
            return False
    
    def get_table_info(self, table_name):
        """
        Get information about a table's schema.
        
        Args:
            table_name (str): Name of the table
            
        Returns:
            pandas.DataFrame: Table schema information
        """
        try:
            if self.db_type == 'sqlite':
                query = f"PRAGMA table_info({table_name})"
                return self.run_query(query)
            
            elif self.db_type == 'mysql':
                query = f"DESCRIBE {table_name}"
                return self.run_query(query)
            
            return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"Error getting schema for table {table_name}: {str(e)}")
            return pd.DataFrame()
    
    def get_table_count(self, table_name, condition=None):
        """
        Get the number of rows in a table.
        
        Args:
            table_name (str): Name of the table
            condition (str, optional): WHERE condition
            
        Returns:
            int: Number of rows
        """
        try:
            query = f"SELECT COUNT(*) as count FROM {table_name}"
            
            if condition:
                query += f" WHERE {condition}"
            
            result = self.run_query(query)
            
            if not result.empty:
                return result.iloc[0]['count']
            
            return 0
        
        except Exception as e:
            logger.error(f"Error getting row count for table {table_name}: {str(e)}")
            return 0
    
    def get_latest_data(self, table_name, date_column, limit=10):
        """
        Get the latest data from a table based on a date column.
        
        Args:
            table_name (str): Name of the table
            date_column (str): Name of the date column
            limit (int): Maximum number of rows to return
            
        Returns:
            pandas.DataFrame: Latest data
        """
        try:
            query = f"SELECT * FROM {table_name} ORDER BY {date_column} DESC LIMIT {limit}"
            return self.run_query(query)
        
        except Exception as e:
            logger.error(f"Error getting latest data from table {table_name}: {str(e)}")
            return pd.DataFrame()
    
    def backup_database(self, backup_path=None):
        """
        Create a backup of the database.
        
        Args:
            backup_path (str, optional): Path to save the backup
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.db_type == 'sqlite':
            if not backup_path:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_dir = os.path.dirname(self.db_path)
                backup_name = f"{os.path.basename(self.db_path)}.{timestamp}.bak"
                backup_path = os.path.join(backup_dir, backup_name)
            
            try:
                # Make sure connection is closed first
                self.close()
                
                # Create backup directory if it doesn't exist
                os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                
                # Use the sqlite3 CLI to create a backup
                import shutil
                shutil.copy2(self.db_path, backup_path)
                
                logger.info(f"Created SQLite database backup at: {backup_path}")
                return True
            
            except Exception as e:
                logger.error(f"Error creating SQLite database backup: {str(e)}")
                return False
        
        elif self.db_type == 'mysql':
            if not backup_path:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_dir = os.path.expanduser('~/.marketmind/backups')
                backup_name = f"{self.mysql_config['database']}.{timestamp}.sql"
                backup_path = os.path.join(backup_dir, backup_name)
            
            try:
                # Make sure directory exists
                os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                
                # Use the mysqldump utility to create a backup
                import subprocess
                cmd = [
                    'mysqldump',
                    f"--host={self.mysql_config['host']}",
                    f"--port={self.mysql_config['port']}",
                    f"--user={self.mysql_config['user']}",
                    f"--password={self.mysql_config['password']}",
                    self.mysql_config['database']
                ]
                
                with open(backup_path, 'w') as f:
                    subprocess.run(cmd, stdout=f, check=True)
                
                logger.info(f"Created MySQL database backup at: {backup_path}")
                return True
            
            except Exception as e:
                logger.error(f"Error creating MySQL database backup: {str(e)}")
                return False
        
        return False 