from pathlib import Path
from ts_trove.eda.univaritate_eda import UnivariateEDA
from plotly.subplots import make_subplots
import plotly.graph_objects as go

class UnivaritateEDAReport:
    def __init__(self, univariate_eda: UnivariateEDA):
        self.univariate_eda = univariate_eda
        self.index_description = self.univariate_eda.describe_time_index()

    def generate_report(self, output_path: Path) -> None:
        """
        Generate a univariate EDA report and save it to the specified output path.

        Parameters:
        output_path (Path): The file path where the report will be saved.
        """
        # Example: Save time series plot
        fig = self.univariate_eda.plot_time_series()
        fig.write_html(output_path / "time_series_plot.html")

        # Additional report generation logic can be added here

    def _ts_and_distribution_panel(self) -> None:
        """
        Plot a univariate time series DataFrame using Plotly.

        Parameters:
        ts (pd.Series): Series with a datetime index and representing the time series data.

        Returns:
        plotly.graph_objs._figure.Figure: A Plotly figure object displaying the time series.
        """

        subplot_grid = make_subplots(rows=2, cols=3,
                            specs=[[{"type": "scatter"}, {"type": "xy"}, {"type": "histogram"}],
                                [{"type": "scatter"}, {"type": "xy"}, {"type": "histogram"}]],
                            subplot_titles=[
                                'Time Series',
                                '',
                                'Distribution Histogram',
                                'Differenced Time Series',
                                '',
                                'Differenced Distribution Histogram'
                            ],
                            shared_yaxes=True,
                            shared_xaxes=True,
                            horizontal_spacing=0.01,
                            column_widths=[0.6, 0.1, 0.3])
        
        # -- time series (1,1) --
        time_series_plot = self.univariate_eda.plot_time_series()
        time_series_plot.data[0].update(line=dict(color='darkblue'))
        subplot_grid.add_trace(time_series_plot.data[0], row=1, col=1)

        # -- box plot (1,2) --
        subplot_grid.add_trace(
            go.Box(y=self.univariate_eda.ts, marker=dict(color='darkblue'), name=''), row=1, col=2
        )

        # -- distribution histogram (1,3) --
        distribution_histogram = self.univariate_eda.plot_distribution_histogram(orientation='h')
        distribution_histogram.data[0].update(marker=dict(color='darkblue'))
        subplot_grid.add_trace(distribution_histogram.data[0], row=1, col=3)

        # -- differenced time series and distribution (2,1) --
        diff_ts_plot = self.univariate_eda.plot_time_series(differenced=True)
        diff_ts_plot.data[0].update(line=dict(color='lightblue'))
        subplot_grid.add_trace(diff_ts_plot.data[0], row=2, col=1)

        # -- differenced box plot (2,2) --
        subplot_grid.add_trace(
            go.Box(y=self.univariate_eda.ts.diff().dropna(), marker=dict(color='lightblue'), name=''), row=2, col=2
        )

        # -- differenced distribution histogram (2,3) --
        differenced_distribution_histogram = self.univariate_eda.plot_distribution_histogram(differenced=True, orientation='h')
        differenced_distribution_histogram.data[0].update(marker=dict(color='lightblue'))
        subplot_grid.add_trace(differenced_distribution_histogram.data[0], row=2, col=3)
        
        return subplot_grid
    
    def _self_correlation_panel(self) -> None:
        """
        Plot ACF and PACF panels.

        Returns:
        plotly.graph_objs._figure.Figure: A Plotly figure object displaying the ACF and PACF.
        """

        subplot_grid = make_subplots(rows=2, cols=2,
                            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                                   [{"type": "scatter"}, {"type": "scatter"}]],
                            subplot_titles=[
                                'Autocorrelation Function (ACF)',
                                'Partial Autocorrelation Function (PACF)'
                                'Autocorrelation Function (ACF) - Resampled',
                                'Partial Autocorrelation Function (PACF) - Resampled'
                            ],
                            horizontal_spacing=0.15,
                            column_widths=[0.5, 0.5])
        
        inferred_freq = self.index_description.get('inferred_frequency', 'h')
        if inferred_freq == 'M':
            lag1, lag2 = 12, 60
            resample_to = 'h'  
        elif inferred_freq == 'h':
            lag1, lag2 = 24, 168
            resample_to = 'd'
        elif inferred_freq == 'd':
            lag1, lag2 = 30, 365
            resample_to = 'w'


        # -- ACF (1,1) --
        acf_plot = self.univariate_eda.plot_acf(nlags=lag1)
        acf_plot.data[0].update(marker=dict(color='darkblue'))
        subplot_grid.add_trace(acf_plot.data[0], row=1, col=1)

        # -- PACF (1,2) --
        pacf_plot = self.univariate_eda.plot_pacf(nlags=lag1)
        pacf_plot.data[0].update(marker=dict(color='darkblue'))
        subplot_grid.add_trace(pacf_plot.data[0], row=1, col=2)

        # -- ACF Resampled (2,1) --
        acf_rs_plot = self.univariate_eda.plot_acf(nlags=lag2, resample_to=resample_to)
        acf_rs_plot.data[0].update(marker=dict(color='darkblue'))
        subplot_grid.add_trace(acf_rs_plot.data[0], row=2, col=1)

        # -- PACF Resampled (2,2) --
        pacf_rs_plot = self.univariate_eda.plot_pacf(nlags=lag2, resample_to=resample_to)
        pacf_rs_plot.data[0].update(marker=dict(color='darkblue'))
        subplot_grid.add_trace(pacf_rs_plot.data[0], row=2, col=2)

        return subplot_grid
    
    def generate(self, output_path: Path) -> None:
        """
        Generate the univariate EDA report and save it to the specified output path.

        Parameters:
        output_path (Path): The file path where the report will be saved.
        """

        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate and save time series and distribution panel
        ts_dist_panel = self._ts_and_distribution_panel()
        ts_dist_panel.write_html(output_path / "ts_and_distribution_panel.html")

        # Generate and save self-correlation panel
        self_corr_panel = self._self_correlation_panel()
        self_corr_panel.write_html(output_path / "self_correlation_panel.html")

        # concat the panels into a single report

        with open(output_path / "univariate_eda_report.html", "w") as report_file:
            report_file.write("<html><head><title>Univariate EDA Report</title></head><body>\n")
            report_file.write("<h1>Time Series and Distribution Panel</h1>\n")
            report_file.write('<iframe src="ts_and_distribution_panel.html" width="100%" height="800"></iframe>\n')
            report_file.write("<h1>Self-Correlation Panel</h1>\n")
            report_file.write('<iframe src="self_correlation_panel.html" width="100%" height="400"></iframe>\n')
            report_file.write("</body></html>\n")