# ts-trove

A project for understanding, implementing, and testing various time series techniques in forecasting, anomaly detection, and time series classification.

## Overview

ts-trove is a comprehensive collection of time series analysis techniques organized into three main categories:

- **Forecasting**: Predicting future values based on historical observations
- **Anomaly Detection**: Identifying unusual patterns or outliers in time series data
- **Classification**: Assigning time series instances to predefined categories

## Project Structure

```
ts-trove/
├── data/                           # Data storage directory
├── notebooks/                      # Jupyter notebooks for experimentation
│   ├── forecasting.ipynb          # Forecasting techniques
│   ├── anomaly_detection.ipynb    # Anomaly detection techniques
│   └── classification.ipynb       # Classification techniques
├── src/
│   └── ts_trove/                  # Main package
│       ├── forecasting/           # Forecasting module
│       │   ├── __init__.py
│       │   └── base.py           # BaseForecaster ABC
│       ├── anomaly_detection/     # Anomaly detection module
│       │   ├── __init__.py
│       │   └── base.py           # BaseAnomalyDetector ABC
│       └── classification/        # Classification module
│           ├── __init__.py
│           └── base.py           # BaseClassifier ABC
└── pyproject.toml                 # Project configuration
```

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate     # On Windows
```

## Usage

### Notebooks

Explore the notebooks in the `notebooks/` directory to learn about different time series techniques:

```bash
jupyter notebook notebooks/
```

### Using the Base Classes

All techniques inherit from abstract base classes that define a common interface:

```python
from ts_trove.forecasting import BaseForecaster
from ts_trove.anomaly_detection import BaseAnomalyDetector
from ts_trove.classification import BaseClassifier
```

## Techniques Roadmap

### Forecasting
- Statistical Methods (ARIMA, SARIMA, Exponential Smoothing)
- Machine Learning Methods (Random Forests, Gradient Boosting, SVR)
- Deep Learning Methods (LSTM, GRU, Transformers)
- Hybrid Methods (Prophet, NeuralProphet)

### Anomaly Detection
- Statistical Methods (Z-Score, Modified Z-Score, IQR)
- Distance-Based Methods (KNN, LOF)
- Density-Based Methods (DBSCAN, Isolation Forest, One-Class SVM)
- Deep Learning Methods (Autoencoders, LSTM Autoencoders, VAE)
- Specialized TS Methods (STL Decomposition, ARIMA Residuals)

### Classification
- Feature-Based Methods (Statistical features, tsfresh, Catch22)
- Distance-Based Methods (DTW-KNN, Euclidean Distance)
- Dictionary-Based Methods (BOSS, cBOSS, WEASEL)
- Shapelet-Based Methods (Shapelet Transform)
- Deep Learning Methods (CNN, ResNet, InceptionTime)
- Ensemble Methods (HIVE-COTE, TS-CHIEF)

## Dependencies

- numpy
- pandas
- matplotlib
- scikit-learn
- jupyter
- notebook

## Contributing

This is a learning and experimentation project. Feel free to add new techniques or improve existing implementations.

## License

MIT
