"""
Evaluation Module

This module handles model evaluation and backtesting:
- Performance metrics calculation
- Backtesting framework
- Statistical analysis
- Risk metrics
"""

from .metrics import MetricsCalculator
from .backtesting import BacktestEngine
from .risk_metrics import RiskAnalyzer
from .statistical_tests import StatisticalTester

__all__ = [
    'MetricsCalculator',
    'BacktestEngine',
    'RiskAnalyzer',
    'StatisticalTester'
] 