import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object

try:
	from xgboost import XGBRegressor  # type: ignore
	exgboost_available = True
except Exception:
	XGBRegressor = None  # type: ignore
	exgboost_available = False

try:
	from catboost import CatBoostRegressor  # type: ignore
	catboost_available = True
except Exception:
	CatBoostRegressor = None  # type: ignore
	catboost_available = False


@dataclass
class ModelTrainerConfig:
	trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
	def __init__(self) -> None:
		self.model_trainer_config = ModelTrainerConfig()

	def _get_models_and_params(self) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
		enable_heavy_models = os.getenv("ENABLE_HEAVY_MODELS", "0") == "1"

		models: Dict[str, Any] = {
			"LinearRegression": LinearRegression(),
			"DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
			"RandomForestRegressor": RandomForestRegressor(random_state=42),
			"GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
			"AdaBoostRegressor": AdaBoostRegressor(random_state=42),
			"KNeighborsRegressor": KNeighborsRegressor(),
		}

		if enable_heavy_models and exgboost_available and XGBRegressor is not None:
			models["XGBRegressor"] = XGBRegressor(random_state=42, objective="reg:squarederror")
		elif enable_heavy_models:
			logging.info("XGBoost is unavailable; skipping XGBRegressor")

		if enable_heavy_models and catboost_available and CatBoostRegressor is not None:
			models["CatBoostRegressor"] = CatBoostRegressor(verbose=False, random_state=42)
		elif enable_heavy_models:
			logging.info("CatBoost is unavailable; skipping CatBoostRegressor")

		if not enable_heavy_models:
			logging.info("Heavy models disabled. Set ENABLE_HEAVY_MODELS=1 to include XGB/CatBoost")

		params: Dict[str, Dict[str, Any]] = {
			"LinearRegression": {},
			"DecisionTreeRegressor": {
				"criterion": ["squared_error", "friedman_mse"],
				"max_features": [None, "sqrt", "log2"],
				"max_depth": [None, 10, 20],
				"min_samples_split": [2, 5],
			},
			"RandomForestRegressor": {
				"n_estimators": [50, 100],
				"max_features": ["sqrt", "log2", None],
				"max_depth": [None, 10],
			},
			"GradientBoostingRegressor": {
				"n_estimators": [100, 150],
				"learning_rate": [0.05, 0.1],
				"subsample": [0.8, 1.0],
			},
			"AdaBoostRegressor": {
				"n_estimators": [50, 100],
				"learning_rate": [0.05, 0.1, 0.5],
			},
			"KNeighborsRegressor": {
				"n_neighbors": [3, 5, 7],
				"weights": ["uniform", "distance"],
				"metric": ["minkowski"],
			},
		}

		if "XGBRegressor" in models:
			params["XGBRegressor"] = {
				"n_estimators": [100, 200],
				"learning_rate": [0.05, 0.1],
				"max_depth": [3, 6],
			}

		if "CatBoostRegressor" in models:
			params["CatBoostRegressor"] = {
				"depth": [4, 6],
				"learning_rate": [0.03, 0.1],
				"iterations": [200],
			}

		return models, params

	def initiate_model_trainer(
		self,
		train_array: NDArray[Any],
		test_array: NDArray[Any],
		preprocessor_path: str | None = None,
	) -> float:
		try:
			logging.info("Split training and test input data")
			train_array = np.asarray(train_array)
			test_array = np.asarray(test_array)

			X_train, y_train, X_test, y_test = (
				train_array[:, :-1],
				train_array[:, -1],
				test_array[:, :-1],
				test_array[:, -1],
			)

			models, params = self._get_models_and_params()
			logging.info("Model training started with %d candidate models", len(models))
			if preprocessor_path:
				logging.info("Preprocessor path received: %s", preprocessor_path)

			model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)

			best_model_name = max(model_report.items(), key=lambda item: item[1])[0]
			best_model_score = model_report[best_model_name]
			best_model = models[best_model_name]

			if best_model_score < 0.6:
				raise ValueError("No best model found with acceptable score")

			save_object(
				file_path=self.model_trainer_config.trained_model_file_path,
				obj=best_model,
			)

			predicted = best_model.predict(X_test)
			r2_square = r2_score(y_test, predicted)
			logging.info("Best model selected: %s", best_model_name)
			logging.info("Best model test score from report: %.4f", best_model_score)
			logging.info("R2 score on held-out test set: %.4f", r2_square)

			return r2_square
		except Exception as e:
			raise CustomException(e, sys) from e

