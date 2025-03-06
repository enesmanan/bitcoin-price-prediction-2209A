from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing.model_preprocessor import FinancialDataPreprocessor


class BitcoinPricePredictor:
    def __init__(
        self,
        target_column: str = "Price",
        feature_columns: List[str] = None,
        time_limit: int = 600,
        eval_metric: str = "root_mean_squared_error",
    ):
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.time_limit = time_limit
        self.eval_metric = eval_metric
        self.predictor = None
        self.feature_cols: Optional[List[str]] = None
        self.preprocessor = FinancialDataPreprocessor(lookback_period=30)
        self.train_data = None
        self.test_data = None

    def prepare_data(
        self,
        data_path: str,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Veriyi hazırlar ve train-test olarak ayırır
        """
        # Veriyi oku
        df = pd.read_csv(data_path)
        df["Date"] = pd.to_datetime(df["Date"])

        # Preprocessor'u kullanarak feature'ları oluştur
        processed_data = self.preprocessor.prepare_data(
            df, train_start, train_end, test_start, test_end
        )

        train_data = processed_data["train"]
        test_data = processed_data["test"]

        # Target değişkenini oluştur (bir sonraki günün kapanış fiyatı)
        train_data["target"] = train_data["Price"].shift(-1)
        test_data["target"] = test_data["Price"].shift(-1)

        # NaN değerleri kaldır
        train_data = train_data.dropna()
        test_data = test_data.dropna()

        # Feature kolonlarını belirle
        exclude_cols = [
            "Date",
            "target",
            "Price",
            "Open",
            "High",
            "Low",
            "Vol.",
            "Change %",
        ]
        self.feature_cols = [
            col for col in train_data.columns if col not in exclude_cols
        ]

        self.train_data = train_data
        self.test_data = test_data

        print(f"Train veri boyutu: {train_data.shape}")
        print(f"Test veri boyutu: {test_data.shape}")
        print(f"Kullanılan feature sayısı: {len(self.feature_cols)}")

        return train_data, test_data

    def train_model(self, train_data: pd.DataFrame) -> None:
        features = (
            self.feature_columns
            if self.feature_columns
            else [
                col
                for col in train_data.columns
                if col not in [self.target_column, "Date"]
            ]
        )

        train_data = train_data[features + [self.target_column]].copy()

        self.predictor = TabularPredictor(
            label=self.target_column,
            eval_metric=self.eval_metric,
            path="model_weights/autogluon",
        )

        self.predictor.fit(
            train_data=train_data,
            time_limit=self.time_limit,
            presets="high_quality",
            num_gpus=1,
        )

    def evaluate_model(self, test_data: pd.DataFrame) -> Dict:
        if self.predictor is None:
            raise ValueError("Model has not been trained yet")

        features = (
            self.feature_columns
            if self.feature_columns
            else [
                col
                for col in test_data.columns
                if col not in [self.target_column, "Date"]
            ]
        )

        test_features = test_data[features].copy()
        predictions = self.predictor.predict(test_features)

        # Metrikleri hesapla
        metrics = self.predictor.evaluate_predictions(
            y_true=test_data[self.target_column],
            y_pred=predictions,
            auxiliary_metrics=True,
        )

        # Feature importance
        try:
            importance = self.predictor.feature_importance(test_features)
        except:
            importance = pd.Series(0, index=features)

        return {
            "metrics": metrics,
            "feature_importance": importance,
            "predictions": pd.DataFrame(
                {"actual": test_data[self.target_column], "predicted": predictions}
            ),
        }

    def make_predictions(self, data: pd.DataFrame) -> np.ndarray:
        """
        Yeni veriler için tahmin yapar
        Args:
            data: Tahmin yapılacak veri
        Returns:
            np.ndarray: Tahminler
        """
        if self.predictor is None:
            raise ValueError("Model henüz eğitilmedi!")

        return self.predictor.predict(data[self.feature_cols])

    def save_model(self, path: str = "bitcoin_price_model") -> None:
        if self.predictor is None:
            raise ValueError("Model henüz eğitilmedi!")
        self.predictor.save(path)

    def load_model(self, path: str = "bitcoin_price_model") -> None:
        self.predictor = TabularPredictor.load(path)
