import sys
import os

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


def run_train_pipeline(csv_path: str, target_column_name: str) -> float:
	try:
		logging.info("Training pipeline started")

		data_ingestion = DataIngestion()
		train_data, test_data = data_ingestion.initiate_data_ingestion(csv_path)

		data_transformation = DataTransformation()
		train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
			train_data, test_data, target_column_name
		)

		model_trainer = ModelTrainer()
		r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)

		logging.info("Training pipeline completed with R2 score: %s", r2_score)
		return r2_score
	except Exception as e:
		raise CustomException(e, sys) from e


if __name__ == "__main__":
	dataset_path = os.path.join("notebooks", "stud.csv")
	score = run_train_pipeline(dataset_path, "math_score")
	print(f"Training completed. R2 score: {score:.4f}")

