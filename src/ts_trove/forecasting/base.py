"""Base class for time series forecasting models."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class BaseForecaster(ABC):
    """Abstract base class for time series forecasting models.

    All forecasting techniques should inherit from this class and implement
    the required abstract methods.
    """

    def __init__(self):
        """Initialize the forecaster."""
        self.is_fitted = False
        self.params: dict[str, Any] = {}

    @abstractmethod
    def fit(self, y: np.ndarray | pd.Series, X: np.ndarray | pd.DataFrame | None = None) -> "BaseForecaster":
        """Fit the forecasting model to training data.

        Args:
            y: Target time series values
            X: Optional exogenous variables

        Returns:
            self: The fitted forecaster instance
        """
        pass

    @abstractmethod
    def predict(self, steps: int, X: np.ndarray | pd.DataFrame | None = None) -> np.ndarray:
        """Generate forecasts for future time steps.

        Args:
            steps: Number of steps ahead to forecast
            X: Optional exogenous variables for forecast period

        Returns:
            Forecasted values
        """
        pass

    def fit_predict(self, y: np.ndarray | pd.Series, steps: int,
                    X: np.ndarray | pd.DataFrame | None = None) -> np.ndarray:
        """Fit the model and generate forecasts.

        Args:
            y: Target time series values
            steps: Number of steps ahead to forecast
            X: Optional exogenous variables

        Returns:
            Forecasted values
        """
        self.fit(y, X)
        return self.predict(steps, X)

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Get parameters of the forecaster.

        Returns:
            Dictionary of model parameters
        """
        pass

    @abstractmethod
    def set_params(self, **params: Any) -> "BaseForecaster":
        """Set parameters of the forecaster.

        Args:
            **params: Model parameters to set

        Returns:
            self: The forecaster instance
        """
        pass
