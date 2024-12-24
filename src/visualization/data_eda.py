from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots


class FinancialDataEDA:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.data["Month"] = self.data["Date"].dt.strftime("%Y-%m")
        self.data["Year"] = self.data["Date"].dt.year
        self._prepare_correlation_data()

    def _prepare_correlation_data(self):
        self.correlation_columns = {
            "Bitcoin Price": "Price",
            "Bitcoin Open": "Open",
            "Bitcoin High": "High",
            "Bitcoin Low": "Low",
            "Bitcoin Change %": "Change %",
            "USD Buy Rate": "usd_buy",
            "USD Sell Rate": "usd_sell",
            "Gold Price": "gold_Price",
            "Gold Open": "gold_Open",
            "Gold High": "gold_High",
            "Gold Low": "gold_Low",
            "Gold Change": "gold_Change",
        }
        self.corr_df = self.data[list(self.correlation_columns.values())]
        self.correlation_matrix = self.corr_df.corr()

    def plot_time_series(self) -> go.Figure:
        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=("Bitcoin Price", "USD/TRY Rate", "Gold Price"),
            vertical_spacing=0.1,
            shared_xaxes=True,
        )

        fig.add_trace(
            go.Scatter(
                x=self.data["Date"],
                y=self.data["Price"],
                name="Bitcoin Price",
                line=dict(color="#f2a900"),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=self.data["Date"],
                y=self.data["usd_buy"],
                name="USD Buy Rate",
                line=dict(color="#2ecc71"),
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=self.data["Date"],
                y=self.data["gold_Price"],
                name="Gold Price",
                line=dict(color="#f1c40f"),
            ),
            row=3,
            col=1,
        )

        fig.update_layout(
            height=900,
            title_text="Bitcoin, USD/TRY and Gold Price Time Series",
            showlegend=True,
            template="plotly_white",
        )

        fig.update_xaxes(rangeslider_visible=False)
        fig.update_yaxes(title_text="USD", row=1, col=1)
        fig.update_yaxes(title_text="TRY", row=2, col=1)
        fig.update_yaxes(title_text="USD", row=3, col=1)

        return fig

    def plot_correlation_matrix(self) -> go.Figure:
        fig = go.Figure(
            data=go.Heatmap(
                z=self.correlation_matrix,
                x=list(self.correlation_columns.keys()),
                y=list(self.correlation_columns.keys()),
                colorscale="RdBu_r",
                zmin=-1,
                zmax=1,
                text=np.round(self.correlation_matrix, 2),
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False,
                colorbar=dict(title="Correlation", titleside="right"),
            )
        )

        fig.update_layout(
            title={
                "text": "Correlation Matrix Heatmap",
                "y": 0.95,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
                "font": dict(size=16),
            },
            width=1000,
            height=1000,
            template="plotly_white",
            xaxis={"side": "bottom"},
            yaxis={"side": "left"},
        )

        # Etiketlerin döndürülmesi
        fig.update_xaxes(tickangle=90)
        fig.update_yaxes(tickangle=0)

        return fig

    def get_detailed_correlations(self) -> pd.DataFrame:
        corr_matrix = self.correlation_matrix.copy()
        significant_corr = pd.DataFrame(columns=["Asset 1", "Asset 2", "Correlation"])

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                significant_corr = pd.concat(
                    [
                        significant_corr,
                        pd.DataFrame(
                            {
                                "Asset 1": [list(self.correlation_columns.keys())[i]],
                                "Asset 2": [list(self.correlation_columns.keys())[j]],
                                "Correlation": [corr_matrix.iloc[i, j]],
                            }
                        ),
                    ]
                )

        return significant_corr.sort_values("Correlation", ascending=False)

    def plot_returns_distribution(self) -> go.Figure:
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Bitcoin Returns",
                "USD Returns",
                "Gold Returns",
                "Normalized QQ Plot",
            ),
        )

        fig.add_trace(
            go.Histogram(
                x=self.data["Change %"],
                name="BTC Returns",
                nbinsx=50,
                marker_color="#f2a900",
            ),
            row=1,
            col=1,
        )

        usd_returns = self.data["usd_buy"].pct_change() * 100
        fig.add_trace(
            go.Histogram(
                x=usd_returns, name="USD Returns", nbinsx=50, marker_color="#2ecc71"
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Histogram(
                x=self.data["gold_Change"],
                name="Gold Returns",
                nbinsx=50,
                marker_color="#f1c40f",
            ),
            row=2,
            col=1,
        )

        btc_returns_sorted = sorted(self.data["Change %"].dropna())
        theoretical_quantiles = pd.Series(
            np.random.normal(0, 1, len(btc_returns_sorted))
        )
        theoretical_quantiles = sorted(theoretical_quantiles)

        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=btc_returns_sorted,
                mode="markers",
                name="QQ Plot",
                marker=dict(color="#f2a900"),
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            height=800, title="Returns Distribution Analysis", template="plotly_white"
        )
        return fig

    def generate_risk_metrics(self) -> Dict:
        metrics = {}

        # Basic statistics for each asset
        for name, col in [
            ("Bitcoin", "Price"),
            ("USD", "usd_buy"),
            ("Gold", "gold_Price"),
        ]:
            metrics[name] = {
                "mean": self.data[col].mean(),
                "std": self.data[col].std(),
                "min": self.data[col].min(),
                "max": self.data[col].max(),
                "sharpe_ratio": self._calculate_sharpe_ratio(col),
                "max_drawdown": self._calculate_drawdown(col),
            }

            if col == "Price":
                metrics[name]["value_at_risk"] = self._calculate_var("Change %")

        return metrics

    def _calculate_sharpe_ratio(
        self, column: str, risk_free_rate: float = 0.02
    ) -> float:
        returns = self.data[column].pct_change()
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / returns.std()

    def _calculate_drawdown(self, column: str) -> float:
        price = self.data[column]
        rolling_max = price.cummax()
        drawdown = (price - rolling_max) / rolling_max
        return drawdown.min() * 100

    def _calculate_var(self, column: str, confidence: float = 0.95) -> float:
        return self.data[column].quantile(1 - confidence)
