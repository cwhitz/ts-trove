"""Base class for time series anomaly detection models."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class BaseAnomalyDetector(ABC):
    """Abstract base class for time series anomaly detection models.

    All anomaly detection techniques should inherit from this class and implement
    the required abstract methods.
    """

    def __init__(self):
        """Initialize the anomaly detector."""
        self.is_fitted = False
        self.params: dict[str, Any] = {}

    @abstractmethod
    def fit(self, X: np.ndarray | pd.DataFrame) -> "BaseAnomalyDetector":
        """Fit the anomaly detection model to training data.

        Args:
            X: Time series data (can be univariate or multivariate)

        Returns:
            self: The fitted detector instance
        """
        pass

    @abstractmethod
    def detect(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Detect anomalies in time series data.

        Args:
            X: Time series data to check for anomalies

        Returns:
            Binary array where 1 indicates anomaly, 0 indicates normal
        """
        pass

    @abstractmethod
    def score(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Compute anomaly scores for time series data.

        Args:
            X: Time series data

        Returns:
            Anomaly scores (higher scores indicate more anomalous points)
        """
        pass

    def fit_detect(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Fit the model and detect anomalies.

        Args:
            X: Time series data

        Returns:
            Binary array where 1 indicates anomaly, 0 indicates normal
        """
        self.fit(X)
        return self.detect(X)

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Get parameters of the anomaly detector.

        Returns:
            Dictionary of model parameters
        """
        pass

    @abstractmethod
    def set_params(self, **params: Any) -> "BaseAnomalyDetector":
        """Set parameters of the anomaly detector.

        Args:
            **params: Model parameters to set

        Returns:
            self: The detector instance
        """
        pass
