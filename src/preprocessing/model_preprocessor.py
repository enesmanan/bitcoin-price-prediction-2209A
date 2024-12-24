from typing import Dict, Tuple

import numpy as np
import pandas as pd


class FinancialDataPreprocessor:

    def __init__(self, lookback_period: int = 30):
        """
        Args:
            lookback_period: Feature hesaplamaları için gereken geçmiş veri miktarı
        """
        self.lookback_period = lookback_period

    def prepare_data(
        self,
        df: pd.DataFrame,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
    ) -> Dict[str, pd.DataFrame]:
        """
        Veriyi hazırlar ve feature'ları oluşturur.

        Args:
            df: Ham veri seti
            train_start: Training başlangıç tarihi
            train_end: Training bitiş tarihi
            test_start: Test başlangıç tarihi
            test_end: Test bitiş tarihi

        Returns:
            Dict[str, pd.DataFrame]: Train ve test dataframe'leri
        """
        # 1. İlk olarak train-test split yap
        train_df, test_df = self._split_time_series_data(
            df, train_start, train_end, test_start, test_end
        )

        # 2. Feature'ları oluştur
        train_features, test_features = self._create_features_for_train_test(
            train_df, test_df
        )

        return {"train": train_features, "test": test_features}

    def _split_time_series_data(
        self,
        df: pd.DataFrame,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Train-test split yapar."""
        # Tarihleri datetime'a çevir
        dates = {
            "train_start": pd.to_datetime(train_start),
            "train_end": pd.to_datetime(train_end),
            "test_start": pd.to_datetime(test_start),
            "test_end": pd.to_datetime(test_end),
        }

        # Veriyi böl
        train_df = df[
            (df["Date"] >= dates["train_start"]) & (df["Date"] <= dates["train_end"])
        ].copy()
        test_df = df[
            (df["Date"] >= dates["test_start"]) & (df["Date"] <= dates["test_end"])
        ].copy()

        print(f"Training veri boyutu: {train_df.shape}")
        print(f"Test veri boyutu: {test_df.shape}")

        return train_df, test_df

    def _create_features_for_train_test(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Train ve test setleri için feature'ları oluşturur."""
        # Train verisi için feature'ları oluştur
        train_features = self._create_full_feature_set(train_df)

        # Test verisi için train'in son kısmını da içeren geçici DataFrame oluştur
        temp_test_df = pd.concat(
            [
                train_df.tail(self.lookback_period),  # Train'in son kısmı
                test_df,  # Test verisi
            ]
        ).reset_index(drop=True)

        # Test verisi için feature'ları oluştur
        temp_test_features = self._create_full_feature_set(temp_test_df)

        # Sadece test dönemine ait kısmı al
        test_features = temp_test_features.tail(len(test_df)).reset_index(drop=True)

        return train_features, test_features

    def _create_full_feature_set(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tüm feature'ları oluşturur."""
        df = self._add_technical_indicators(df)
        df = self._add_time_features(df)
        df = self._add_candlestick_patterns(df)
        return df.fillna(method="ffill").fillna(method="bfill")

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Teknik göstergeleri ekler."""
        df_temp = df.copy()

        # 1. RSI hesaplama
        def calculate_rsi(data: pd.Series, periods: int = 14) -> pd.Series:
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        price_shifted = df_temp["Price"].shift(1)
        df_temp["RSI"] = calculate_rsi(price_shifted)

        # 2. Hareketli Ortalamalar
        windows = [7, 14, 21]
        for window in windows:
            df_temp[f"MA_{window}"] = price_shifted.rolling(window=window).mean()
            df_temp[f"EMA_{window}"] = price_shifted.ewm(
                span=window, adjust=False
            ).mean()

        # 3. MACD
        exp1 = price_shifted.ewm(span=12, adjust=False).mean()
        exp2 = price_shifted.ewm(span=26, adjust=False).mean()
        df_temp["MACD"] = exp1 - exp2
        df_temp["MACD_Signal"] = df_temp["MACD"].ewm(span=9, adjust=False).mean()

        # 4. Bollinger Bands
        window = 20
        df_temp["BB_middle"] = price_shifted.rolling(window=window).mean()
        df_temp["BB_upper"] = (
            df_temp["BB_middle"] + 2 * price_shifted.rolling(window=window).std()
        )
        df_temp["BB_lower"] = (
            df_temp["BB_middle"] - 2 * price_shifted.rolling(window=window).std()
        )

        # 5. Fiyat Değişim Oranları
        for window in [1, 3, 7]:
            df_temp[f"ROC_{window}"] = price_shifted.pct_change(periods=window) * 100

        # 6. Volatilite
        df_temp["Volatility"] = price_shifted.rolling(window=14).std()

        # 7. True Range ve ATR
        high_shifted = df_temp["High"].shift(1)
        low_shifted = df_temp["Low"].shift(1)

        df_temp["TR"] = np.maximum(
            high_shifted - low_shifted,
            np.maximum(
                abs(high_shifted - price_shifted), abs(low_shifted - price_shifted)
            ),
        )
        df_temp["ATR"] = df_temp["TR"].rolling(window=14).mean()

        # 8. Momentum
        df_temp["Momentum"] = price_shifted - price_shifted.shift(4)

        # 9. Hacim göstergeleri
        volume_shifted = df_temp["Vol."].shift(1)
        df_temp["Volume_MA"] = volume_shifted.rolling(window=14).mean()
        df_temp["Volume_ROC"] = volume_shifted.pct_change() * 100

        # 10. Cross-asset göstergeler
        df_temp["BTC_Gold_Ratio"] = price_shifted / df_temp["gold_Price"].shift(1)
        df_temp["BTC_USD_Ratio"] = price_shifted / df_temp["usd_buy"].shift(1)

        return df_temp

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Zaman bazlı özellikleri ekler."""
        df_temp = df.copy()
        df_temp["Day_of_Week"] = df_temp["Date"].dt.dayofweek
        df_temp["Month"] = df_temp["Date"].dt.month
        df_temp["Quarter"] = df_temp["Date"].dt.quarter
        df_temp["Year"] = df_temp["Date"].dt.year
        df_temp["Day_of_Year"] = df_temp["Date"].dt.dayofyear
        df_temp["Is_Weekend"] = df_temp["Date"].dt.dayofweek.isin([5, 6]).astype(int)
        return df_temp

    def _add_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Candlestick pattern'ları ekler."""
        df_temp = df.copy()

        open_shifted = df_temp["Open"].shift(1)
        price_shifted = df_temp["Price"].shift(1)
        high_shifted = df_temp["High"].shift(1)
        low_shifted = df_temp["Low"].shift(1)

        df_temp["Body_Size"] = abs(open_shifted - price_shifted)
        df_temp["Upper_Shadow"] = high_shifted - df_temp[["Open", "Price"]].shift(
            1
        ).max(axis=1)
        df_temp["Lower_Shadow"] = (
            df_temp[["Open", "Price"]].shift(1).min(axis=1) - low_shifted
        )

        return df_temp

    def check_data_quality(
        self, train_data: pd.DataFrame, test_data: pd.DataFrame
    ) -> None:
        print("\nVeri Kalitesi Raporu:")
        print("-" * 50)

        print("\nTrain veri boyutu:", train_data.shape)
        print("Test veri boyutu:", test_data.shape)

        print("\nTrain seti eksik değer sayıları:")
        print(train_data.isnull().sum()[train_data.isnull().sum() > 0])

        print("\nTest seti eksik değer sayıları:")
        print(test_data.isnull().sum()[test_data.isnull().sum() > 0])

        print("\nTrain seti tarih aralığı:")
        print(f"Başlangıç: {train_data['Date'].min()}")
        print(f"Bitiş: {train_data['Date'].max()}")

        print("\nTest seti tarih aralığı:")
        print(f"Başlangıç: {test_data['Date'].min()}")
        print(f"Bitiş: {test_data['Date'].max()}")
