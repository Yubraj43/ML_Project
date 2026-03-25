import os
import sys
from typing import Any, Dict

import dill
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path: str, obj: object) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys) from e


def load_object(file_path: str) -> Any:
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys) from e


def evaluate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models: Dict[str, Any],
    params: Dict[str, Dict[str, Any]],
) -> Dict[str, float]:
    try:
        report: Dict[str, float] = {}

        for model_name, model in models.items():
            param_grid = params.get(model_name, {})

            if param_grid:
                grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
                grid.fit(X_train, y_train)
                model.set_params(**grid.best_params_)

            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_score
            _ = train_score

        return report
    except Exception as e:
        raise CustomException(e, sys) from e
