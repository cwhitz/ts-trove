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
                                'Differenced Histogram'
                            ],
                            shared_yaxes=True,
                            shared_xaxes=True,
                            horizontal_spacing=0.01,
                            vertical_spacing=0.1,
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
        
        # hide legend
        subplot_grid.update_layout(showlegend=False, margin=dict(t=50, b=20, l=20, r=20))

        return subplot_grid
    
    def _windowed_statistics_panel(self) -> None:
        """
        Plot rolling mean and rolling standard deviation.

        Returns:
        plotly.graph_objs._figure.Figure: A Plotly figure object displaying the rolling statistics.
        """

        rolling_stats_plot = self.univariate_eda.plot_rolling_statistics()
        rolling_stats_plot.update_layout(margin=dict(t=50, b=20, l=20, r=20))

        return rolling_stats_plot
    
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
                                'ACF',
                                'PACF',
                                'ACF - Resampled',
                                'PACF - Resampled'
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

        subplot_grid.update_layout(showlegend=False, margin=dict(t=50, b=20, l=20, r=20))

        return subplot_grid
    
    def _write_html_table(self, d: dict) -> str:
        rows = "".join([f"<tr><td><strong>{k}</strong></td><td>{v}</td></tr>" for k, v in d.items()])
        return f"""
        <div class="table-container">
            <table>
                <thead>
                    <tr><th>Metric</th><th>Value</th></tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
            """
        
    def generate(self, output_path: Path) -> None:
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get content
        tsi_html = self._write_html_table(self.univariate_eda.describe_time_index())
        stationarity_html = self._write_html_table(self.univariate_eda.describe_stationarity())
        
        # Get Plotly HTML but extract only the <div> and <script> parts (full_html=False)
        ts_dist = self._ts_and_distribution_panel().to_html(full_html=False, include_plotlyjs='cdn')
        windowed = self._windowed_statistics_panel().to_html(full_html=False, include_plotlyjs=False)
        self_corr = self._self_correlation_panel().to_html(full_html=False, include_plotlyjs=False)

        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Univariate EDA Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7f6; color: #333; margin: 0; padding: 40px; }}
                .container {{ max-width: 1200px; margin: auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-top: 40px; }}
                .table-container {{ margin-bottom: 30px; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
                th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #3498db; color: white; }}
                tr:hover {{ background-color: #f1f1f1; }}
                .plot-card {{ background: #fff; border: 1px solid #eee; border-radius: 4px; margin-bottom: 20px; padding: 10px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Univariate EDA Report</h1>
                
                <h3>Time Index Description</h3>
                {tsi_html}

                <h1>Time Series & Distribution</h1>
                <div class="plot-card">{ts_dist}</div>

                <h3>Stationarity Test Results</h3>
                {stationarity_html}

                <h1>Windowed Statistics</h1>
                <div class="plot-card">{windowed}</div>

                <h1>Self-Correlation Analysis</h1>
                <div class="plot-card">{self_corr}</div>
            </div>
        </body>
        </html>
        """

        with open(output_path / "univariate_eda_report.html", "w") as f:
            f.write(html_content)