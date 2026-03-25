import os
import sys
from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
	train_data_path: str = os.path.join("artifacts", "train.csv")
	test_data_path: str = os.path.join("artifacts", "test.csv")
	raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
	def __init__(self) -> None:
		self.ingestion_config = DataIngestionConfig()

	def initiate_data_ingestion(self, csv_path: str) -> Tuple[str, str]:
		logging.info("Entered the data ingestion method")
		try:
			if not os.path.exists(csv_path):
				raise FileNotFoundError(f"Input data file not found: {csv_path}")

			df = pd.read_csv(csv_path)
			logging.info("Read dataset as dataframe with shape %s", df.shape)

			os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

			df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

			train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
			train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
			test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

			logging.info("Ingestion completed")
			return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

		except Exception as e:
			raise CustomException(e, sys) from e


if __name__ == "__main__":
	try:
		from src.components.data_transformation import DataTransformation
		from src.components.model_trainer import ModelTrainer

		csv_path = os.path.join("notebooks", "stud.csv")

		data_ingestion = DataIngestion()
		train_data_path, test_data_path = data_ingestion.initiate_data_ingestion(csv_path)

		data_transformation = DataTransformation()
		train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
			train_data_path,
			test_data_path,
			"math_score",
		)

		model_trainer = ModelTrainer()
		r2_square = model_trainer.initiate_model_trainer(
			train_arr,
			test_arr,
			preprocessor_path,
		)

		print(f"Train array shape: {train_arr.shape}")
		print(f"Test array shape: {test_arr.shape}")
		print(f"Preprocessor saved at: {preprocessor_path}")
		print(f"R2 score after hyperparameter tuning: {r2_square:.4f}")
	except Exception as e:
		raise CustomException(e, sys) from e

