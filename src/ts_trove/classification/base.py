"""Base class for time series classification models."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class BaseClassifier(ABC):
    """Abstract base class for time series classification models.

    All time series classification techniques should inherit from this class
    and implement the required abstract methods.
    """

    def __init__(self):
        """Initialize the classifier."""
        self.is_fitted = False
        self.params: dict[str, Any] = {}
        self.classes_: np.ndarray | None = None

    @abstractmethod
    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> "BaseClassifier":
        """Fit the classification model to training data.

        Args:
            X: Time series data (each row is a time series instance)
            y: Target labels

        Returns:
            self: The fitted classifier instance
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict class labels for time series data.

        Args:
            X: Time series data to classify

        Returns:
            Predicted class labels
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for time series data.

        Args:
            X: Time series data to classify

        Returns:
            Class probabilities (shape: n_samples x n_classes)
        """
        pass

    def fit_predict(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> np.ndarray:
        """Fit the model and predict class labels.

        Args:
            X: Time series data
            y: Target labels

        Returns:
            Predicted class labels for X
        """
        self.fit(X, y)
        return self.predict(X)

    @abstractmethod
    def score(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> float:
        """Compute accuracy score on test data.

        Args:
            X: Time series data
            y: True labels

        Returns:
            Accuracy score
        """
        pass

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Get parameters of the classifier.

        Returns:
            Dictionary of model parameters
        """
        pass

    @abstractmethod
    def set_params(self, **params: Any) -> "BaseClassifier":
        """Set parameters of the classifier.

        Args:
            **params: Model parameters to set

        Returns:
            self: The classifier instance
        """
        pass
