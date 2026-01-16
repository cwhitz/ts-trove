import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.stattools import pacf, adfuller, acf
from plotly.subplots import make_subplots
from pathlib import Path

class UnivariateEDA:
    def __init__(self, ts: pd.Series):
        self.ts = ts

    def describe_time_index(self) -> dict:
        """
        Describe the time index of a DataFrame in terms of frequency, time span, and missing timestamps.

        Parameters:
        ts (pd.Series): Series with a datetime index.
        
        Returns:
        dict: A dictionary containing the inferred frequency, start time, end time, and missing timestamps
        of the time index.

        Raises:
        ValueError: If the DataFrame index is not a datetime type.
        """

        index = self.ts.index
        if not pd.api.types.is_datetime64_any_dtype(index):
            raise ValueError("DataFrame index must be a datetime type.")

        # Frequency
        inferred_freq = pd.infer_freq(index)
        
        # Time span
        start_time = index.min()
        end_time = index.max()
        
        # Missing timestamps
        full_time_index = pd.date_range(start=start_time, end=end_time, freq=inferred_freq)
        missing_timestamps = full_time_index.difference(index)
        
        return {
            'inferred_frequency': inferred_freq,
            'start_time': start_time,
            'end_time': end_time,
            'n_missing_timestamps': len(missing_timestamps),
            'missing_timestamps': missing_timestamps.tolist()
        }

    def plot_time_series(self, differenced: bool = False) -> go.Figure:
        """
        Plot a univariate time series DataFrame using Plotly.

        Parameters:
        differenced (bool): Whether to plot the differenced time series.

        Returns:
        fig: Plotly figure object displaying the time series.
        """

        if differenced:
            ts_to_plot = self.ts.diff().dropna()
        else:
            ts_to_plot = self.ts

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts_to_plot.index, y=ts_to_plot, mode='lines', name=ts_to_plot.name))
        fig.update_layout(title=ts_to_plot.name + ' time series',
                        xaxis_title='Time')
        fig.update_layout(template='plotly_white')

        return fig
    
    def describe_distribution(self) -> dict:
        """
        Describe the distribution of a time series.

        Parameters:
        ts (pd.Series): A pandas Series representing the time series data.

        Returns:
        dict: A dictionary containing key statistics of the distribution.
        """

        return {

            'min': self.ts.min(),
            '25th_percentile': self.ts.quantile(0.25),
            '50th_percentile': self.ts.quantile(0.50),
            '75th_percentile': self.ts.quantile(0.75),
            'max': self.ts.max(),
            'range': self.ts.max() - self.ts.min(),
            'mean': self.ts.mean(),
            'std_dev': self.ts.std(),
            'skewness': self.ts.skew(),
            'kurtosis': self.ts.kurtosis()
        }
    
    def make_box_plot(self, differenced: bool = False) -> go.Figure:
        """
        Create a box plot of the time series data.

        Parameters:
        ts (pd.Series): A pandas Series representing the time series data.

        Returns:
        fig: Plotly figure object displaying the box plot.
        """
        if differenced:
            ts_to_plot = self.ts.diff().dropna()
        else:
            ts_to_plot = self.ts

        fig = go.Figure()
        fig.add_trace(go.Box(y=ts_to_plot, name=ts_to_plot.name, boxmean='sd', marker=dict(color='darkblue')))
        fig.update_layout(title='Box Plot of ' + ts_to_plot.name,
                        yaxis_title=ts_to_plot.name)
        fig.update_layout(template='plotly_white')

        return fig
    
    def plot_rolling_statistics(self, window: int = 24) -> go.Figure:
        rolling_mean = self.ts.rolling(window=window).mean()
        rolling_std = self.ts.rolling(window=window).std()

        fig = go.Figure()
        
        # Primary Y-Axis (y1)
        fig.add_trace(go.Scatter(x=self.ts.index, y=self.ts, mode='lines', yaxis='y1', opacity=0.7, line=dict(color='lightblue'), name=self.ts.name))
        fig.add_trace(go.Scatter(x=rolling_mean.index, y=rolling_mean, yaxis='y1', mode='lines', line=dict(color='blue', width=1), name='Rolling Mean'))
        
        # Secondary Y-Axis (y2)
        fig.add_trace(go.Scatter(x=rolling_std.index, y=rolling_std, mode='lines', line={'color': 'black', 'width': 0.5},yaxis='y2', opacity=0.5, name='Rolling Std Dev'))

        fig.update_layout(
            title=f'{self.ts.name} with Rolling Statistics',
            xaxis_title='Time',
            yaxis=dict(
                title=self.ts.name,
            ),
            yaxis2=dict(
                title="Rolling Std Dev",
                anchor="x",
                overlaying="y", # This is crucial: it overlays the first y-axis
                side="right"    # Put it on the right side
            ),
            template='plotly_white',
            legend=dict(x=1.1, y=1) # Move legend so it doesn't block the secondary axis
        )

        return fig

    def plot_distribution_histogram(self, differenced: bool = False, orientation: str = 'v') -> go.Figure:
        """
        Plot a histogram of the data.

        Parameters:
        orientation (str): Orientation of the histogram, 'v' for vertical or 'h' for horizontal.

        Returns:
        fig: Plotly figure object displaying the histogram.
        """

        if differenced:
            ts_to_plot = self.ts.diff().dropna()
            self.ts = ts_to_plot
        else:
            ts_to_plot = self.ts

        fig = go.Figure()
        if orientation == 'h':
            fig.add_trace(go.Histogram(y=ts_to_plot, nbinsy=50))
            fig.update_layout(title='Distribution of ' + ts_to_plot.name,
                            yaxis_title=ts_to_plot.name,
                            xaxis_title='Count')
        elif orientation == 'v':
            fig.add_trace(go.Histogram(x=ts_to_plot, nbinsx=50))
            fig.update_layout(title='Distribution of ' + ts_to_plot.name,
                            xaxis_title=ts_to_plot.name,
                            yaxis_title='Count')
        fig.update_layout(template='plotly_white')

        return fig
    
    def describe_stationarity(self) -> dict:
        """
        Perform the Augmented Dickey-Fuller test to assess the stationarity of a time series.

        Parameters:
        ts (pd.Series): A pandas Series representing the time series data.

        Returns:
        dict: A dictionary containing the ADF test results.
        """

        adf_result = adfuller(self.ts)
        adf_dict ={
            'adf_statistic': adf_result[0],
            'p_value': adf_result[1],
            'used_lag': adf_result[2],
            'n_obs': adf_result[3],
            'ic_best': adf_result[5]
        }

        for key, value in adf_result[4].items():
            adf_dict[f'critical_value_{key}'] = value

        if adf_dict['p_value'] < adf_dict['critical_value_5%']:
            adf_dict['stationarity'] = True
            return adf_dict
        else:
            adf_dict['stationarity'] = False
            return adf_dict
        
    def describe_acf(self, nlags: int = 24) -> dict:
        """
        Describe the autocorrelation function (ACF) of a time series.

        Parameters:
        ts (pd.Series): A pandas Series representing the time series data.
        nlags (int): Number of lags to compute the ACF for.

        Returns:
        dict: A dictionary containing the ACF values for each lag.
        """

        acf_values = acf(self.ts, nlags=nlags)
        acf_dict = {f'lag_{i}': acf_values[i] for i in range(len(acf_values))}
        return acf_dict

    def plot_acf(self, nlags: int = 24, resample_to: str = None) -> go.Figure:
        """
        Plot the autocorrelation function (ACF) of a time series.

        Parameters:
        nlags (int): Number of lags to include in the ACF plot.

        Returns:
        fig: Plotly figure object displaying the ACF.
        """

        if resample_to:
            ts_to_use = self.ts.resample(resample_to).mean().dropna()
        else:
            ts_to_use = self.ts

        acf_values = acf(ts_to_use, nlags=nlags)
        
        critical_value = 1.96 / (len(ts_to_use) ** 0.5)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(range(len(acf_values))), y=acf_values))
        fig.add_shape(type='line',
                        x0=0, y0=critical_value, x1=nlags, y1=critical_value,
                        line=dict(color='Red', dash='dash'))
        fig.add_shape(type='line',
                        x0=0, y0=-critical_value, x1=nlags, y1=-critical_value,
                        line=dict(color='Red', dash='dash'))
        
        fig.update_layout(title='Autocorrelation Function (ACF)',
                        xaxis_title='Lags',
                        yaxis_title='ACF')
        fig.update_layout(template='plotly_white')

        return fig

    def describe_pacf(self, nlags: int = 24) -> dict:
        """
        Describe the partial autocorrelation function (PACF) of a time series.

        Parameters:
        ts (pd.Series): A pandas Series representing the time series data.
        nlags (int): Number of lags to compute the PACF for.

        Returns:
        dict: A dictionary containing the PACF values for each lag.
        """

        pacf_values = pacf(self.ts, nlags=nlags)
        pacf_dict = {f'lag_{i}': pacf_values[i] for i in range(len(pacf_values))}
        return pacf_dict

    def plot_pacf(self, nlags: int = 24, resample_to: str = None) -> go.Figure:
        """
        Plot the partial autocorrelation function (PACF) of a time series.

        Parameters:
        nlags (int): Number of lags to include in the PACF plot.
        resample_to (str): Resampling frequency if needed.

        Returns:
        fig: Plotly figure object displaying the PACF.
        """

        if resample_to:
            ts_to_use = self.ts.resample(resample_to).mean().dropna()
        else:
            ts_to_use = self.ts

        pacf_values = pacf(ts_to_use, nlags=nlags)

        critical_value = 1.96 / (len(ts_to_use) ** 0.5)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(range(len(pacf_values))), y=pacf_values))
        fig.add_shape(type='line',
                        x0=0, y0=critical_value, x1=nlags, y1=critical_value,
                        line=dict(color='Red', dash='dash'))
        fig.add_shape(type='line',
                        x0=0, y0=-critical_value, x1=nlags, y1=-critical_value,
                        line=dict(color='Red', dash='dash'))
        fig.update_layout(title='Partial Autocorrelation Function (PACF)',
                        xaxis_title='Lags',
                        yaxis_title='PACF')
        fig.update_layout(template='plotly_white')

        return fig