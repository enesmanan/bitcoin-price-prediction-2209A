from typing import Dict

import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


class FinancialLSTM:
    def __init__(
        self,
        sequence_length: int = 20,
        epochs: int = 150,
        batch_size: int = 32,
        lstm_units: list = [128, 64, 32],
        dropout_rate: float = 0.1,
    ):
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None

    def build_model(self, input_shape: tuple):
        inputs = tf.keras.Input(shape=input_shape)

        # First Bidirectional LSTM layer
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                self.lstm_units[0],
                return_sequences=True,
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
            )
        )(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)

        # Middle LSTM layers
        for units in self.lstm_units[1:-1]:
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    units,
                    return_sequences=True,
                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
                )
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)

        # Final LSTM layer
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                self.lstm_units[-1],
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
            )
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)

        # Dense layers with skip connections
        skip = x
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Add()([x, skip])

        x = tf.keras.layers.Dense(32, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # Output layer
        outputs = tf.keras.layers.Dense(1)(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Custom loss function - düzeltilmiş versiyon
        def custom_loss(y_true, y_pred):
            mse = tf.reduce_mean(tf.square(y_true - y_pred))
            mae = tf.reduce_mean(tf.abs(y_true - y_pred))
            return 0.7 * mse + 0.3 * mae

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=custom_loss,
            metrics=["mae"],
        )

        self.model = model
        return model

    def get_callbacks(self):
        return [
            EarlyStopping(
                monitor="val_loss", patience=25, restore_best_weights=True, mode="min"
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.2, patience=10, min_lr=0.00001, mode="min"
            ),
        ]

    def train(self, data: Dict) -> Dict:
        """Train the LSTM model and plot training history"""
        X_train, y_train = data["train"]["X"], data["train"]["y"]
        X_test, y_test = data["test"]["X"], data["test"]["y"]

        if self.model is None:
            self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))

        # Train the model
        self.history = self.model.fit(
            X_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_test, y_test),
            callbacks=self.get_callbacks(),
            verbose=1,
        )

        # Plot training history
        self.plot_training_history()

        return {
            "train_loss": self.history.history["loss"][-1],
            "train_mae": self.history.history["mae"][-1],
            "val_loss": self.history.history["val_loss"][-1],
            "val_mae": self.history.history["val_mae"][-1],
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the trained model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(X, verbose=0)

    def plot_training_history(self):
        """Plot training history including loss and metrics"""
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Model Loss", "Model MAE"),
            vertical_spacing=0.2,
        )

        # Plot loss
        fig.add_trace(
            go.Scatter(
                y=self.history.history["loss"],
                name="Training Loss",
                line=dict(color="blue", width=2),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                y=self.history.history["val_loss"],
                name="Validation Loss",
                line=dict(color="red", width=2),
            ),
            row=1,
            col=1,
        )

        # Plot MAE
        fig.add_trace(
            go.Scatter(
                y=self.history.history["mae"],
                name="Training MAE",
                line=dict(color="blue", width=2),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                y=self.history.history["val_mae"],
                name="Validation MAE",
                line=dict(color="red", width=2),
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            height=800,
            width=1000,
            title_text="Training History",
            showlegend=True,
            template="plotly_white",
        )

        # Update axes labels
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="MAE", row=2, col=1)
        fig.update_xaxes(title_text="Epoch", row=2, col=1)

        fig.show()

    def plot_predictions(self, dates, y_true, y_pred):
        """Plot actual vs predicted values"""
        fig = go.Figure()

        # Add actual values
        fig.add_trace(
            go.Scatter(
                x=dates, y=y_true, name="Actual", line=dict(color="blue", width=2)
            )
        )

        # Add predicted values
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=y_pred,
                name="Predicted",
                line=dict(color="red", width=2, dash="dash"),
            )
        )

        # Calculate error bands
        error = np.abs(y_true - y_pred)
        error_band = np.std(error) * 2

        # Add error bands
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=y_pred + error_band,
                fill=None,
                mode="lines",
                line_color="rgba(255,0,0,0)",
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=y_pred - error_band,
                fill="tonexty",
                mode="lines",
                line_color="rgba(255,0,0,0)",
                fillcolor="rgba(255,0,0,0.2)",
                showlegend=False,
            )
        )

        fig.update_layout(
            title="Bitcoin Price - Actual vs Predicted",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode="x unified",
            showlegend=True,
            template="plotly_white",
            height=600,
            width=1000,
            xaxis=dict(
                tickformat="%Y-%m-%d",
                tickangle=45,
            ),
            yaxis=dict(tickformat="$,.0f"),
        )
        fig.show()

    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate and return evaluation metrics"""
        return {
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "r2": r2_score(y_true, y_pred),
            "mape": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        }
