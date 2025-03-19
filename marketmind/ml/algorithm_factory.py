"""
Algorithm Factory Module

This module provides a factory for creating and configuring machine learning algorithms.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

# Initialize logger
logger = logging.getLogger('marketmind.ml.algorithm_factory')

class AlgorithmFactory:
    """
    Factory for creating and configuring machine learning algorithms.
    """
    
    def __init__(self, config=None):
        """
        Initialize the AlgorithmFactory with configuration.
        
        Args:
            config (dict): Configuration dictionary containing algorithm settings.
        """
        from .. import load_config
        self.config = config or load_config()
        
        # Get default parameters from config
        self.default_params = self.config.get('ml', {}).get('default_params', {})
        
        # Define supported algorithms
        self.supported_regressors = {
            'linear': LinearRegression,
            'ridge': Ridge,
            'lasso': Lasso,
            'elastic_net': ElasticNet,
            'random_forest': RandomForestRegressor,
            'gradient_boosting': GradientBoostingRegressor,
            'svr': SVR,
            'knn': KNeighborsRegressor,
            'xgboost': XGBRegressor,
            'lightgbm': LGBMRegressor
        }
        
        self.supported_classifiers = {
            'logistic': LogisticRegression,
            'random_forest': RandomForestClassifier,
            'gradient_boosting': GradientBoostingClassifier,
            'svc': SVC,
            'knn': KNeighborsClassifier,
            'xgboost': XGBClassifier,
            'lightgbm': LGBMClassifier
        }
    
    def get_regressor(self, algorithm='random_forest', params=None):
        """
        Get a regression algorithm instance.
        
        Args:
            algorithm (str): Name of the algorithm
            params (dict, optional): Algorithm parameters
            
        Returns:
            object: Initialized algorithm instance
        """
        if algorithm not in self.supported_regressors:
            logger.error(f"Unsupported regression algorithm: {algorithm}")
            logger.info(f"Supported regression algorithms: {', '.join(self.supported_regressors.keys())}")
            raise ValueError(f"Unsupported regression algorithm: {algorithm}")
        
        # Get algorithm class
        algo_class = self.supported_regressors[algorithm]
        
        # Get default parameters for this algorithm
        default_params = self.default_params.get('regressors', {}).get(algorithm, {})
        
        # Merge with provided parameters
        if params:
            merged_params = {**default_params, **params}
        else:
            merged_params = default_params
        
        logger.info(f"Creating {algorithm} regressor with parameters: {merged_params}")
        
        # Create and return algorithm instance
        return algo_class(**merged_params)
    
    def get_classifier(self, algorithm='random_forest', params=None):
        """
        Get a classification algorithm instance.
        
        Args:
            algorithm (str): Name of the algorithm
            params (dict, optional): Algorithm parameters
            
        Returns:
            object: Initialized algorithm instance
        """
        if algorithm not in self.supported_classifiers:
            logger.error(f"Unsupported classification algorithm: {algorithm}")
            logger.info(f"Supported classification algorithms: {', '.join(self.supported_classifiers.keys())}")
            raise ValueError(f"Unsupported classification algorithm: {algorithm}")
        
        # Get algorithm class
        algo_class = self.supported_classifiers[algorithm]
        
        # Get default parameters for this algorithm
        default_params = self.default_params.get('classifiers', {}).get(algorithm, {})
        
        # Merge with provided parameters
        if params:
            merged_params = {**default_params, **params}
        else:
            merged_params = default_params
        
        logger.info(f"Creating {algorithm} classifier with parameters: {merged_params}")
        
        # Create and return algorithm instance
        return algo_class(**merged_params)
    
    def get_param_grid(self, algorithm, model_type='regressor'):
        """
        Get parameter grid for hyperparameter tuning.
        
        Args:
            algorithm (str): Name of the algorithm
            model_type (str): Type of model ('regressor' or 'classifier')
            
        Returns:
            dict: Parameter grid for GridSearchCV
        """
        # Get parameter grid from config
        if model_type == 'regressor':
            param_grid = self.config.get('ml', {}).get('param_grids', {}).get('regressors', {}).get(algorithm, {})
        else:
            param_grid = self.config.get('ml', {}).get('param_grids', {}).get('classifiers', {}).get(algorithm, {})
        
        if not param_grid:
            logger.warning(f"No parameter grid found for {algorithm} {model_type}, using empty grid")
            return {}
        
        return param_grid
    
    def tune_algorithm(self, X, y, algorithm, model_type='regressor', cv=5, scoring=None, params=None):
        """
        Tune algorithm hyperparameters using grid search.
        
        Args:
            X (array-like): Feature matrix
            y (array-like): Target vector
            algorithm (str): Name of the algorithm
            model_type (str): Type of model ('regressor' or 'classifier')
            cv (int or cross-validation generator): Cross-validation strategy
            scoring (str): Scoring metric for model evaluation
            params (dict, optional): Additional parameters to include in grid search
            
        Returns:
            object: Best fitted model
        """
        # Get base model
        if model_type == 'regressor':
            model = self.get_regressor(algorithm)
            if scoring is None:
                scoring = 'neg_mean_squared_error'
        else:
            model = self.get_classifier(algorithm)
            if scoring is None:
                scoring = 'f1'
        
        # Get parameter grid
        param_grid = self.get_param_grid(algorithm, model_type)
        
        # Merge with provided parameters
        if params:
            for key, values in params.items():
                param_grid[key] = values
        
        if not param_grid:
            logger.warning(f"No parameters to tune for {algorithm} {model_type}, returning default model")
            return model
        
        # Use TimeSeriesSplit for time series data if specified
        if isinstance(cv, str) and cv == 'time_series':
            cv = TimeSeriesSplit(n_splits=5)
        
        # Run grid search
        logger.info(f"Tuning {algorithm} {model_type} with parameter grid: {param_grid}")
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        
        # Log best parameters
        logger.info(f"Best parameters for {algorithm} {model_type}: {grid_search.best_params_}")
        logger.info(f"Best score for {algorithm} {model_type}: {grid_search.best_score_}")
        
        return grid_search.best_estimator_
    
    def evaluate_regressor(self, model, X_test, y_test):
        """
        Evaluate a regression model.
        
        Args:
            model: Trained model
            X_test (array-like): Test feature matrix
            y_test (array-like): Test target vector
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        y_pred = model.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        logger.info(f"Regression metrics: {metrics}")
        return metrics
    
    def evaluate_classifier(self, model, X_test, y_test):
        """
        Evaluate a classification model.
        
        Args:
            model: Trained model
            X_test (array-like): Test feature matrix
            y_test (array-like): Test target vector
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        # Add ROC AUC for binary classification
        if len(np.unique(y_test)) == 2:
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
                metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
            except:
                logger.warning("Could not calculate ROC AUC score")
        
        logger.info(f"Classification metrics: {metrics}")
        return metrics
    
    def train_ensemble(self, X, y, algorithms=None, model_type='regressor', weights=None):
        """
        Train an ensemble of algorithms.
        
        Args:
            X (array-like): Feature matrix
            y (array-like): Target vector
            algorithms (list): List of algorithm names
            model_type (str): Type of model ('regressor' or 'classifier')
            weights (list, optional): Weights for each algorithm
            
        Returns:
            tuple: (List of fitted models, weight vector)
        """
        if not algorithms:
            if model_type == 'regressor':
                # Use a default set of regressors
                algorithms = ['random_forest', 'gradient_boosting', 'xgboost']
            else:
                # Use a default set of classifiers
                algorithms = ['random_forest', 'gradient_boosting', 'xgboost']
        
        # Validate algorithms
        valid_algos = []
        for algo in algorithms:
            if model_type == 'regressor' and algo in self.supported_regressors:
                valid_algos.append(algo)
            elif model_type == 'classifier' and algo in self.supported_classifiers:
                valid_algos.append(algo)
            else:
                logger.warning(f"Skipping unsupported algorithm: {algo}")
        
        if not valid_algos:
            logger.error(f"No valid algorithms for ensemble")
            raise ValueError(f"No valid algorithms for ensemble")
        
        # Initialize models
        models = []
        for algo in valid_algos:
            if model_type == 'regressor':
                model = self.get_regressor(algo)
            else:
                model = self.get_classifier(algo)
            
            models.append(model)
        
        # Train models
        fitted_models = []
        for i, model in enumerate(models):
            logger.info(f"Training {valid_algos[i]} {model_type} for ensemble")
            model.fit(X, y)
            fitted_models.append(model)
        
        # Calculate weights if not provided
        if weights is None:
            # Use equal weights
            weights = np.ones(len(fitted_models)) / len(fitted_models)
        else:
            # Normalize weights to sum to 1
            weights = np.array(weights)
            weights = weights / np.sum(weights)
        
        logger.info(f"Ensemble weights: {weights}")
        
        return fitted_models, weights
    
    def predict_ensemble(self, models, weights, X):
        """
        Make predictions using an ensemble of models.
        
        Args:
            models (list): List of fitted models
            weights (array-like): Weights for each model
            X (array-like): Feature matrix
            
        Returns:
            array: Weighted ensemble predictions
        """
        # Get predictions from each model
        predictions = np.array([model.predict(X) for model in models])
        
        # Apply weights
        weighted_predictions = np.average(predictions, axis=0, weights=weights)
        
        return weighted_predictions
    
    def feature_importance(self, model, feature_names=None):
        """
        Get feature importance from a trained model.
        
        Args:
            model: Trained model
            feature_names (list, optional): List of feature names
            
        Returns:
            pandas.DataFrame: DataFrame with feature importances
        """
        # Check if model has feature_importances_ attribute
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        # Check if model has coef_ attribute (linear models)
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
            if importances.ndim > 1:
                importances = np.mean(importances, axis=0)
        else:
            logger.warning("Model does not have feature importances")
            return pd.DataFrame()
        
        # Create DataFrame
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, model, model_path):
        """
        Save a trained model to disk.
        
        Args:
            model: Trained model
            model_path (str): Path to save the model
            
        Returns:
            bool: True if successful, False otherwise
        """
        import joblib
        
        try:
            joblib.dump(model, model_path)
            logger.info(f"Saved model to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, model_path):
        """
        Load a trained model from disk.
        
        Args:
            model_path (str): Path to the saved model
            
        Returns:
            object: Loaded model
        """
        import joblib
        
        try:
            model = joblib.load(model_path)
            logger.info(f"Loaded model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None
    
    def format_result(self, model_type, model_name, metrics, feature_importance=None, timestamp=None):
        """
        Format model results for storage or display.
        
        Args:
            model_type (str): Type of model ('regressor' or 'classifier')
            model_name (str): Name of the algorithm
            metrics (dict): Evaluation metrics
            feature_importance (pandas.DataFrame, optional): Feature importance
            timestamp (datetime, optional): Timestamp
            
        Returns:
            dict: Formatted results
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        result = {
            'model_type': model_type,
            'model_name': model_name,
            'metrics': metrics,
            'timestamp': timestamp.isoformat()
        }
        
        if feature_importance is not None:
            result['feature_importance'] = feature_importance.to_dict(orient='records')
        
        return result 