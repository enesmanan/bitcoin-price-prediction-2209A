import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ModelVisualizer:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def plot_model_metrics(self):
        model_names = list(self.pipeline.metrics.keys())
        rmse_values = [metrics["rmse"] for metrics in self.pipeline.metrics.values()]
        mae_values = [metrics["mae"] for metrics in self.pipeline.metrics.values()]
        r2_values = [metrics["r2"] for metrics in self.pipeline.metrics.values()]

        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=("RMSE Comparison", "MAE Comparison", "R² Score Comparison"),
            vertical_spacing=0.1,
        )

        fig.add_trace(
            go.Bar(
                x=model_names,
                y=rmse_values,
                name="RMSE",
                text=[f"{val:,.0f}" for val in rmse_values],
                textposition="auto",
                marker_color="rgb(55, 83, 109)",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=model_names,
                y=mae_values,
                name="MAE",
                text=[f"{val:,.0f}" for val in mae_values],
                textposition="auto",
                marker_color="rgb(26, 118, 255)",
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=model_names,
                y=r2_values,
                name="R²",
                text=[f"{val:.4f}" for val in r2_values],
                textposition="auto",
                marker_color="rgb(58, 171, 115)",
            ),
            row=3,
            col=1,
        )

        fig.update_layout(
            height=900,
            showlegend=False,
            title_text="Model Performance Metrics",
            template="plotly_white",
        )

        return fig

    def plot_best_model_predictions(self):
        best_model = min(self.pipeline.metrics.items(), key=lambda x: x[1]["rmse"])[0]
        pred_data = self.pipeline.predictions[best_model]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=pred_data["test_dates"],
                y=pred_data["y_true"],
                name="Actual",
                line=dict(color="rgb(49, 130, 189)", width=2),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=pred_data["test_dates"],
                y=pred_data["y_pred"],
                name="Predicted",
                line=dict(color="rgb(204, 88, 73)", width=2, dash="dash"),
            )
        )

        metrics = self.pipeline.metrics[best_model]
        title = (
            f"Bitcoin Price - Actual vs Predicted ({best_model})<br>"
            + f'RMSE: {metrics["rmse"]:,.0f} | '
            + f'MAE: {metrics["mae"]:,.0f} | '
            + f'R²: {metrics["r2"]:.4f}'
        )

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_white",
            hovermode="x unified",
        )

        return fig

    def plot_all_predictions(self):
        fig = go.Figure()

        first_model = list(self.pipeline.predictions.keys())[0]
        fig.add_trace(
            go.Scatter(
                x=self.pipeline.test_data["Date"],
                y=self.pipeline.predictions[first_model]["y_true"],
                name="Actual",
                line=dict(color="black", width=3),
            )
        )

        colors = px.colors.qualitative.Set3
        for (name, pred), color in zip(self.pipeline.predictions.items(), colors):
            metrics = self.pipeline.metrics[name]
            fig.add_trace(
                go.Scatter(
                    x=pred["test_dates"],
                    y=pred["y_pred"],
                    name=f'{name} (RMSE: {metrics["rmse"]:,.0f})',
                    line=dict(color=color, width=1.5, dash="dash"),
                )
            )

        fig.update_layout(
            title="All Models - Predictions Comparison",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )

        return fig

    def plot_feature_importance(self, model_name="Random_Forest"):
        importance_df = self.pipeline.get_feature_importance(model_name)

        if importance_df is None:
            return None

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=importance_df["importance"],
                y=importance_df["feature"],
                orientation="h",
                marker_color="rgb(26, 118, 255)",
            )
        )

        fig.update_layout(
            title=f"Feature Importance ({model_name})",
            xaxis_title="Importance",
            yaxis_title="Features",
            template="plotly_white",
            height=max(400, len(importance_df) * 30),
        )

        return fig
