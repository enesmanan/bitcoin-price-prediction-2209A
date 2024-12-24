from typing import Dict, Tuple

import numpy as np
import pandas as pd


class FinancialDataPreprocessor:
    def __init__(self):
        self.bitcoin_data = None
        self.usd_data = None
        self.gold_data = None
        self.merged_data = None

    def load_data(self, bitcoin_path: str, usd_path: str, gold_path: str) -> None:
        self.bitcoin_data = pd.read_csv(bitcoin_path)
        self.usd_data = pd.read_csv(usd_path)
        self.gold_data = pd.read_csv(gold_path)

    def format_bitcoin_data(self) -> pd.DataFrame:
        df = self.bitcoin_data.copy()
        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")

        numeric_columns = ["Price", "Open", "High", "Low"]
        for column in numeric_columns:
            df[column] = df[column].str.replace(",", "").astype(float)

        df["Vol."] = (
            df["Vol."]
            .replace({"K": "*1e3", "M": "*1e6", "B": "*1e9"}, regex=True)
            .map(pd.eval)
            .astype(float)
        )

        df["Change %"] = df["Change %"].str.replace("%", "").astype(float)
        return df.sort_values("Date")

    def format_usd_data(self) -> pd.DataFrame:
        df = self.usd_data.copy()
        df["Tarih"] = pd.to_datetime(df["Tarih"], format="%d-%m-%Y")

        numeric_columns = ["TP DK USD A YTL", "TP DK USD S YTL"]
        df[numeric_columns] = df[numeric_columns].fillna(method="ffill")
        df[numeric_columns] = df[numeric_columns].astype(float)

        column_mapping = {
            "Tarih": "Date",
            "TP DK USD A YTL": "usd_buy",
            "TP DK USD S YTL": "usd_sell",
        }
        return df.rename(columns=column_mapping).sort_values("Date")

    def format_gold_data(self) -> pd.DataFrame:
        df = self.gold_data.copy()

        numeric_columns = ["Şimdi", "Açılış", "Yüksek", "Düşük"]
        for col in numeric_columns:
            df[col] = df[col].str.replace(".", "").str.replace(",", ".").astype(float)

        df["Fark %"] = (
            df["Fark %"].str.replace("%", "").str.replace(",", ".").astype(float)
        )
        df["Tarih"] = pd.to_datetime(df["Tarih"], format="%d.%m.%Y")

        df = df.drop(columns=["Hac."], errors="ignore").fillna(method="ffill")

        column_mapping = {
            "Tarih": "Date",
            "Şimdi": "gold_Price",
            "Açılış": "gold_Open",
            "Yüksek": "gold_High",
            "Düşük": "gold_Low",
            "Fark %": "gold_Change",
        }
        return df.rename(columns=column_mapping).sort_values("Date")

    def merge_data(self) -> pd.DataFrame:
        bitcoin_df = self.format_bitcoin_data()
        usd_df = self.format_usd_data()
        gold_df = self.format_gold_data()

        merged_data = pd.merge(bitcoin_df, usd_df, on="Date", how="left")
        merged_data = pd.merge(merged_data, gold_df, on="Date", how="left")
        self.merged_data = merged_data.fillna(method="ffill").sort_values("Date")

        return self.merged_data

    def get_data_info(self) -> Dict:
        if self.merged_data is None:
            raise ValueError("Please run merge_data() first")

        return {
            "shape": self.merged_data.shape,
            "date_range": (
                self.merged_data["Date"].min(),
                self.merged_data["Date"].max(),
            ),
            "unique_dates": self.merged_data["Date"].nunique(),
            "missing_values": self.merged_data.isnull().sum().to_dict(),
        }
