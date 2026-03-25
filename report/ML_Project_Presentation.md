% Student Exam Performance Prediction System
% ML Project Presentation
% March 2026

# Project Overview

- End-to-end ML system to predict student math score
- Converts raw student data into real-time predictions
- Combines training pipeline, model selection, and web serving
- Deployed on Render for cloud access

# Business Problem

- Institutions need early indicators of student performance
- Manual analysis is slow and inconsistent
- Goal: estimate math score from contextual and academic inputs

# Project Objective

- Build a reproducible ML pipeline
- Train and evaluate multiple regression models
- Select the best model based on test $R^2$
- Serve predictions through a simple Flask web app

# Dataset and Features

- Source: notebooks/stud.csv
- Target: math_score
- Features:
- gender
- race_ethnicity
- parental_level_of_education
- lunch
- test_preparation_course
- reading_score
- writing_score

# End-to-End Workflow

1. Data ingestion from CSV
2. Train-test split and artifact creation
3. Data preprocessing pipeline fit and save
4. Multi-model training and tuning
5. Best model selection and persistence
6. Inference through Flask prediction route

# Data Preprocessing

- Numerical pipeline:
- Median imputation
- Standard scaling
- Categorical pipeline:
- Most-frequent imputation
- One-hot encoding
- Standard scaling (sparse-safe)
- Combined using ColumnTransformer

# Model Training Strategy

- Candidate models:
- Linear Regression
- Decision Tree, Random Forest
- Gradient Boosting, AdaBoost
- KNN Regressor
- Optional heavy models:
- XGBoost and CatBoost via ENABLE_HEAVY_MODELS=1
- Hyperparameter tuning with GridSearchCV

# Model Selection Logic

- Evaluate tuned models on test set
- Select model with highest test $R^2$
- Quality gate: fail training if best score < 0.6
- Save selected model artifact to artifacts/model.pkl

# Deployment Architecture

- Flask web app for inference UI
- Prediction endpoint: /predictdata
- Model and preprocessor loaded from artifacts
- Render deployment:
- Start command: gunicorn app:app
- Live URL: https://ml-project-g7m8.onrender.com/

# Deliverables Produced

- Training pipeline and inference pipeline
- Saved artifacts:
- artifacts/preprocessor.pkl
- artifacts/model.pkl
- Web interface templates for input and results
- Runtime logs for traceability

# Strengths and Current Gaps

- Strengths:
- Modular code structure
- Reproducible artifact flow
- Practical model quality threshold
- Gaps:
- Automated tests
- Data validation layer
- Experiment tracking
- CI/CD and containerization

# Next Improvements

1. Add unit and integration tests
2. Add schema and range validation checks
3. Clean and standardize local requirements.txt
4. Add health endpoint and API versioning
5. Add Docker and automated release pipeline

# Conclusion

- Strong portfolio-grade ML application
- Demonstrates full lifecycle from training to deployment
- Ready for extension into production-grade system with testing and CI/CD
