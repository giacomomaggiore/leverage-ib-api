"""Helpers package exports.

Public API:
- Data IO: `load_data`, `common_window`
- Stats: `log_returns`, `covariance`, `covariance_shrunk`, `sharpe`
- Optimization: `min_variance`, `max_sharpe`, `drop_near_duplicates`
"""

from .fetch import load_data, common_window
from .stats import log_returns, covariance, covariance_shrunk, sharpe
from .weights import min_variance, max_sharpe, drop_near_duplicates

__all__ = [
	# data
	"load_data",
	"common_window",
	# stats
	"log_returns",
	"covariance",
	"covariance_shrunk",
	"sharpe",
	# optimization
	"min_variance",
	"max_sharpe",
    "drop_near_duplicates",
]
