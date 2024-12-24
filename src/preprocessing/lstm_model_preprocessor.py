from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class LSTMDataPreprocessor:
    def __init__(
        self,
        sequence_length: int = 20,
        target_column: str = "Price",
        feature_columns: List[str] = None,
    ):
        self.sequence_length = sequence_length
        self.target_column = target_column
        self.feature_columns = feature_columns or [
            "Price",
            "High",
            "Low",
            "Vol.",
            "MA_7",
            "RSI",
            "MACD",
            "BB_middle",
            "Price_Momentum_1",
            "Price_Momentum_3",
        ]
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

    def add_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add price momentum features to the dataset"""
        df = data.copy()

        # Price momentum features
        if "Price_Momentum_1" not in df.columns:
            df["Price_Momentum_1"] = df["Price"].pct_change()
        if "Price_Momentum_3" not in df.columns:
            df["Price_Momentum_3"] = df["Price"].pct_change(3)

        # Volume momentum if volume exists
        if "Vol." in df.columns and "Volume_Momentum_1" not in df.columns:
            df["Volume_Momentum_1"] = df["Vol."].pct_change()

        return df

    def prepare_lstm_data(
        self, preprocessed_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Prepare data for LSTM model"""
        train_data = preprocessed_data["train"]
        test_data = preprocessed_data["test"]

        # Add momentum features if they don't exist
        train_data = self.add_momentum_features(train_data)
        test_data = self.add_momentum_features(test_data)

        # Create sequences
        X_train, y_train = self._prepare_sequences(train_data)
        X_test, y_test = self._prepare_sequences(test_data)

        return {
            "train": {"X": X_train, "y": y_train},
            "test": {"X": X_test, "y": y_test},
        }

    def _prepare_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM"""
        # Fill NaN values
        data = data.fillna(method="ffill").fillna(method="bfill")

        # Get features and target
        features = data[self.feature_columns].values
        target = data[self.target_column].values.reshape(-1, 1)

        # Check data length
        if len(features) <= self.sequence_length:
            raise ValueError(
                f"Data length ({len(features)}) must be greater than sequence_length ({self.sequence_length})"
            )

        # Scale features and target
        if not hasattr(self.feature_scaler, "scale_"):
            features = self.feature_scaler.fit_transform(features)
            target = self.target_scaler.fit_transform(target)
        else:
            features = self.feature_scaler.transform(features)
            target = self.target_scaler.transform(target)

        # Create sequences
        X, y = [], []
        for i in range(len(features) - self.sequence_length):
            X.append(features[i : (i + self.sequence_length)])
            y.append(target[i + self.sequence_length])

        return np.array(X), np.array(y)

    def inverse_transform_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Transform predictions back to original scale"""
        return self.target_scaler.inverse_transform(predictions)
