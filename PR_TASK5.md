# PR: Task 5 - Model Training and Experiment Tracking

## Description
This PR implements the model training pipeline for the Credit Risk Probability Model. It includes data splitting, model training (Logistic Regression and Random Forest), hyperparameter tuning, and experiment tracking using MLflow.

## Changes
- **src/train.py**: 
  - Implemented data loading from processed CSV.
  - Added `train_model` function to train and evaluate models.
  - Integrated MLflow for tracking parameters, metrics, and models.
  - Implemented hyperparameter tuning for Random Forest using GridSearchCV.
- **tests/test_data_processing.py**: 
  - Added unit tests for data splitting and model training functions.
- **requirements.txt**: 
  - Added `mlflow` and `pytest` dependencies.

## Key Features
- **Experiment Tracking**: All runs are logged to MLflow, including accuracy, AUC, and model artifacts.
- **Model Comparison**: Logistic Regression and Random Forest models were trained and compared.
- **Hyperparameter Tuning**: Optimized Random Forest parameters for better performance.

## How to Test
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run Training Pipeline**:
   ```bash
   python src/train.py
   ```
3. **Run Tests**:
   ```bash
   pytest tests/test_data_processing.py
   ```
4. **View MLflow UI**:
   ```bash
   mlflow ui
   ```
   Open `http://127.0.0.1:5000` to view experiments.

## Results
- **Logistic Regression**: Accuracy ~0.91, AUC ~0.92
- **Random Forest (Best)**: Accuracy ~0.99, AUC ~0.99

## Checklist
- [x] Code compiles and runs without errors.
- [x] Unit tests passed.
- [x] MLflow tracking is functional.
- [x] Code is formatted and clean.
