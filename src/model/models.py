import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy.stats import randint, uniform
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from tabulate import tabulate
from xgboost import XGBRegressor


class FinancialModelPipeline:
    def __init__(self, train_data, test_data, target_col="Price", scale_target=True):
        self.train_data = train_data.copy()
        self.test_data = test_data.copy()
        self.target_col = target_col
        self.scale_target = scale_target
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler() if scale_target else None
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        self.tuning_results = {}
        self.initialize_models()

    def prepare_features(self):
        self.feature_cols = [
            col
            for col in self.train_data.columns
            if col not in [self.target_col, "Date"]
        ]

        X_train = self.feature_scaler.fit_transform(self.train_data[self.feature_cols])
        X_test = self.feature_scaler.transform(self.test_data[self.feature_cols])

        if self.scale_target:
            y_train = self.target_scaler.fit_transform(
                self.train_data[[self.target_col]]
            )
            y_test = self.target_scaler.transform(self.test_data[[self.target_col]])
            y_train = y_train.ravel()
            y_test = y_test.ravel()
        else:
            y_train = self.train_data[self.target_col].values
            y_test = self.test_data[self.target_col].values

        return X_train, X_test, y_train, y_test

    def calculate_mape(self, y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def initialize_models(self):
        self.models = {
            "Linear_Regression": LinearRegression(),
            "Decision_Tree": DecisionTreeRegressor(random_state=42),
            "SVR": SVR(kernel="rbf", C=1.0),
            "Random_Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
            "LightGBM": LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
            "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
            "CatBoost": CatBoostRegressor(iterations=100, random_state=42, verbose=0),
        }

    def get_param_distributions(self):
        return {
            "Random_Forest": {
                "n_estimators": randint(100, 500),
                "max_depth": randint(5, 30),
                "min_samples_split": randint(2, 20),
                "min_samples_leaf": randint(1, 10),
            },
            "XGBoost": {
                "n_estimators": randint(100, 500),
                "max_depth": randint(3, 15),
                "learning_rate": uniform(0.01, 0.3),
                "subsample": uniform(0.6, 0.4),
                "colsample_bytree": uniform(0.6, 0.4),
            },
            "LightGBM": {
                "n_estimators": randint(100, 500),
                "max_depth": randint(3, 15),
                "learning_rate": uniform(0.01, 0.3),
                "subsample": uniform(0.6, 0.4),
                "colsample_bytree": uniform(0.6, 0.4),
            },
            "CatBoost": {
                "iterations": randint(100, 500),
                "depth": randint(4, 10),
                "learning_rate": uniform(0.01, 0.3),
                "l2_leaf_reg": uniform(1, 10),
            },
        }

    def tune_hyperparameters(self, X, y, model_name, n_iter=50):
        if model_name not in self.get_param_distributions():
            return self.models[model_name]

        param_dist = self.get_param_distributions()[model_name]
        base_model = self.models[model_name]

        def custom_rmse_scorer(estimator, X, y):
            y_pred = estimator.predict(X)
            if self.scale_target:
                y = self.target_scaler.inverse_transform(y.reshape(-1, 1)).ravel()
                y_pred = self.target_scaler.inverse_transform(
                    y_pred.reshape(-1, 1)
                ).ravel()
            return -np.sqrt(mean_squared_error(y, y_pred))

        tscv = TimeSeriesSplit(n_splits=5)
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=tscv,
            scoring=custom_rmse_scorer,
            n_jobs=-1,
            random_state=42,
            verbose=0,
        )

        random_search.fit(X, y)
        self.tuning_results[model_name] = {
            "best_params": random_search.best_params_,
            "best_score": -random_search.best_score_,
        }

        return random_search.best_estimator_

    def perform_cross_validation(self, X, y, model):
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]

            model.fit(X_train_cv, y_train_cv)
            y_pred_cv = model.predict(X_val_cv)

            if self.scale_target:
                y_val_orig = self.target_scaler.inverse_transform(
                    y_val_cv.reshape(-1, 1)
                )
                y_pred_orig = self.target_scaler.inverse_transform(
                    y_pred_cv.reshape(-1, 1)
                )
            else:
                y_val_orig, y_pred_orig = y_val_cv, y_pred_cv

            cv_scores.append(
                {
                    "rmse": np.sqrt(mean_squared_error(y_val_orig, y_pred_orig)),
                    "mae": mean_absolute_error(y_val_orig, y_pred_orig),
                    "mape": self.calculate_mape(y_val_orig, y_pred_orig),
                }
            )

        return cv_scores

    def train_and_evaluate(self):
        X_train, X_test, y_train, y_test = self.prepare_features()

        # DataFrame to store results
        results = pd.DataFrame(
            columns=[
                "Model",
                "Scaled RMSE",
                "RMSE",
                "MAE",
                "MAPE",
                "R²",
                "CV RMSE",
                "CV MAE",
                "CV MAPE",
            ]
        )

        for name, model in self.models.items():
            # print(f"\nTraining {name}...")

            model.fit(X_train, y_train)
            y_pred_test = model.predict(X_test)

            scaled_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

            if self.scale_target:
                y_test_orig = self.target_scaler.inverse_transform(
                    y_test.reshape(-1, 1)
                )
                y_pred_test = self.target_scaler.inverse_transform(
                    y_pred_test.reshape(-1, 1)
                )

            test_rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_test))
            test_mae = mean_absolute_error(y_test_orig, y_pred_test)
            test_mape = self.calculate_mape(y_test_orig, y_pred_test)
            test_r2 = r2_score(y_test_orig, y_pred_test)

            cv_scores = self.perform_cross_validation(X_train, y_train, model)
            cv_rmse = np.mean([s["rmse"] for s in cv_scores])
            cv_mae = np.mean([s["mae"] for s in cv_scores])
            cv_mape = np.mean([s["mape"] for s in cv_scores])

            # Add results to DataFrame
            results.loc[len(results)] = [
                name,
                f"{scaled_rmse:.6f}",
                f"{test_rmse:,.0f}",
                f"{test_mae:,.0f}",
                f"{test_mape:.2f}%",
                f"{test_r2:.4f}",
                f"{cv_rmse:,.0f}",
                f"{cv_mae:,.0f}",
                f"{cv_mape:.2f}%",
            ]

            # Store metrics and predictions
            self.metrics[name] = {
                "scaled_rmse": scaled_rmse,
                "rmse": test_rmse,
                "mae": test_mae,
                "mape": test_mape,
                "r2": test_r2,
                "cv_rmse": cv_rmse,
                "cv_mae": cv_mae,
                "cv_mape": cv_mape,
            }

            self.predictions[name] = {
                "test_dates": self.test_data["Date"],
                "y_true": y_test_orig.ravel(),
                "y_pred": y_pred_test.ravel(),
            }

        # Print results table
        print(
            "\n" + tabulate(results, headers="keys", tablefmt="grid", showindex=False)
        )

    def tune_models(self, model_names=None, n_iter=50):
        X_train, _, y_train, _ = self.prepare_features()

        if model_names is None:
            model_names = list(self.get_param_distributions().keys())

        for name in model_names:
            if name in self.get_param_distributions():
                print(f"\nTuning {name}...")
                self.models[name] = self.tune_hyperparameters(
                    X_train, y_train, name, n_iter
                )

    def get_feature_importance(self, model_name):
        if model_name not in self.models:
            return None

        model = self.models[model_name]
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
            feature_imp = pd.DataFrame(
                {"feature": self.feature_cols, "importance": importance}
            )
            return feature_imp.sort_values("importance", ascending=False)
        return None
