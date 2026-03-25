import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd

from src.exception import CustomException
from src.utils import load_object


@dataclass
class PredictPipelineConfig:
	model_path: str = os.path.join("artifacts", "model.pkl")
	preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")


class PredictPipeline:
	def __init__(self) -> None:
		self.config = PredictPipelineConfig()

	def predict(self, features: pd.DataFrame) -> List[float]:
		try:
			model = load_object(self.config.model_path)
			preprocessor = load_object(self.config.preprocessor_path)

			data_scaled = preprocessor.transform(features)
			preds = model.predict(data_scaled)
			return preds.tolist()
		except Exception as e:
			raise CustomException(e, sys) from e


class CustomData:
	def __init__(self, **kwargs: Any) -> None:
		self.data: Dict[str, Any] = kwargs

	def get_data_as_data_frame(self) -> pd.DataFrame:
		try:
			return pd.DataFrame([self.data])
		except Exception as e:
			raise CustomException(e, sys) from e

