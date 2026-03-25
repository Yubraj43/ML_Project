import os
import sys
from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
	preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
	def __init__(self) -> None:
		self.data_transformation_config = DataTransformationConfig()

	def get_data_transformer_object(self) -> ColumnTransformer:
		try:
			numerical_columns: List[str] = ["writing_score", "reading_score"]
			categorical_columns: List[str] = [
				"gender",
				"race_ethnicity",
				"parental_level_of_education",
				"lunch",
				"test_preparation_course",
			]

			num_pipeline = Pipeline(
				steps=[
					("imputer", SimpleImputer(strategy="median")),
					("scaler", StandardScaler()),
				]
			)

			cat_pipeline = Pipeline(
				steps=[
					("imputer", SimpleImputer(strategy="most_frequent")),
					("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
					("scaler", StandardScaler(with_mean=False)),
				]
			)

			logging.info("Numerical columns scaling completed")
			logging.info("Categorical columns encoding completed")

			preprocessor = ColumnTransformer(
				[
					("num_pipeline", num_pipeline, numerical_columns),
					("cat_pipeline", cat_pipeline, categorical_columns),
				]
			)

			return preprocessor
		except Exception as e:
			raise CustomException(e, sys) from e

	def initiate_data_transformation(
		self, train_path: str, test_path: str, target_column_name: str = "math_score"
	) -> Tuple[np.ndarray, np.ndarray, str]:
		try:
			train_df = pd.read_csv(train_path)
			test_df = pd.read_csv(test_path)

			logging.info("Read train and test data completed")
			logging.info("Obtaining preprocessing object")

			if target_column_name not in train_df.columns:
				raise ValueError(f"Target column '{target_column_name}' not found in train data")
			if target_column_name not in test_df.columns:
				raise ValueError(f"Target column '{target_column_name}' not found in test data")

			input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
			target_feature_train_df = train_df[target_column_name]

			input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
			target_feature_test_df = test_df[target_column_name]

			preprocessor_obj = self.get_data_transformer_object()

			logging.info("Applying preprocessing object on training and testing dataframes")

			input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
			input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

			# Convert sparse outputs to dense before concatenating target values.
			train_arr_any: Any = input_feature_train_arr
			test_arr_any: Any = input_feature_test_arr
			if hasattr(train_arr_any, "toarray"):
				input_feature_train_arr = train_arr_any.toarray()
			if hasattr(test_arr_any, "toarray"):
				input_feature_test_arr = test_arr_any.toarray()

			train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
			test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

			save_object(
				file_path=self.data_transformation_config.preprocessor_obj_file_path,
				obj=preprocessor_obj,
			)

			logging.info("Saved preprocessing object")

			return (
				train_arr,
				test_arr,
				self.data_transformation_config.preprocessor_obj_file_path,
			)

		except Exception as e:
			raise CustomException(e, sys) from e

