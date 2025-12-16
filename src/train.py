import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import argparse
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import sys
import os

# Add src to path to import custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import CategoricalEncoder, NumericalScaler, MissingValueImputer

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def prepare_data(df):
    # Drop IDs and non-feature columns
    ids_to_drop = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId', 'TransactionStartTime']
    
    # Check if CurrencyCode and CountryCode are constant
    if df['CurrencyCode'].nunique() == 1:
        ids_to_drop.append('CurrencyCode')
    if df['CountryCode'].nunique() == 1:
        ids_to_drop.append('CountryCode')
        
    df = df.drop(columns=[c for c in ids_to_drop if c in df.columns])
    
    # Handle Categorical Missing Values
    cat_cols = ['ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 'PricingStrategy']
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].fillna('Unknown')
            
    return df

def train_model():
    # Load Data
    data_path = 'data/processed/data_labeled.csv'
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return

    df = load_data(data_path)
    
    # Prepare Data
    # Define Target
    target_col = 'is_high_risk'
    if target_col not in df.columns:
        print(f"Target column {target_col} not found.")
        return

    # Drop IDs but keep categorical features for encoding
    cat_cols = ['ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 'PricingStrategy']
    cat_cols = [c for c in cat_cols if c in df.columns]
    
    # Convert to string to ensure they are treated as categorical
    for c in cat_cols:
        df[c] = df[c].astype(str)
        
    df = prepare_data(df)
    
    # Re-identify columns after drop
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Identify num_cols present in X_train for scaling and imputation
    # Note: CategoricalEncoder will change columns, so we need to be careful with column names in pipeline
    # But MissingValueImputer (custom) selects numeric columns by default.
    
    current_num_cols = X_train.select_dtypes(include=['number']).columns.tolist()
    
    # MLflow Setup
    mlflow.set_experiment("Credit_Risk_Model_Experiment")
    
    # Model 1: Logistic Regression
    with mlflow.start_run(run_name="Logistic_Regression"):
        pipeline_lr = Pipeline([
            ('imputer', MissingValueImputer(strategy='mean', columns=current_num_cols)),
            ('encoder', CategoricalEncoder(method='onehot')), 
            ('scaler', NumericalScaler(method='standard', columns=current_num_cols)),
            ('model', LogisticRegression(max_iter=1000))
        ])
        
        pipeline_lr.fit(X_train, y_train)
        y_pred = pipeline_lr.predict(X_test)
        y_prob = pipeline_lr.predict_proba(X_test)[:, 1]
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        print(f"Logistic Regression - Accuracy: {acc}, AUC: {auc}")
        
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", auc)
        
        mlflow.sklearn.log_model(pipeline_lr, "model")

    # Model 2: Random Forest
    with mlflow.start_run(run_name="Random_Forest"):
        pipeline_rf = Pipeline([
            ('imputer', MissingValueImputer(strategy='mean', columns=current_num_cols)),
            ('encoder', CategoricalEncoder(method='onehot')),
            ('scaler', NumericalScaler(method='standard', columns=current_num_cols)),
            ('model', RandomForestClassifier(random_state=42))
        ])
        
        pipeline_rf.fit(X_train, y_train)
        y_pred = pipeline_rf.predict(X_test)
        y_prob = pipeline_rf.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        print(f"Random Forest - Accuracy: {acc}, AUC: {auc}")
        
        mlflow.log_param("model", "RandomForest")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", auc)
        
        mlflow.sklearn.log_model(pipeline_rf, "model")
        
    # Hyperparameter Tuning for Random Forest
    print("Starting Hyperparameter Tuning for Random Forest...")
    with mlflow.start_run(run_name="Random_Forest_Tuning"):
        pipeline_rf = Pipeline([
            ('imputer', MissingValueImputer(strategy='mean', columns=current_num_cols)),
            ('encoder', CategoricalEncoder(method='onehot')),
            ('scaler', NumericalScaler(method='standard', columns=current_num_cols)),
            ('model', RandomForestClassifier(random_state=42))
        ])
        
        param_grid = {
            'model__n_estimators': [50, 100],
            'model__max_depth': [None, 10],
            'model__min_samples_split': [2, 5]
        }
        
        grid_search = GridSearchCV(pipeline_rf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        print(f"Best RF - Accuracy: {acc}, AUC: {auc}")
        print(f"Best Params: {best_params}")
        
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)
        
        mlflow.sklearn.log_model(best_model, "best_model")
        
        # Save model locally for deployment
        import joblib
        os.makedirs('models', exist_ok=True)
        joblib.dump(best_model, 'models/model.pkl')
        print("Best model saved to models/model.pkl")

if __name__ == "__main__":
    train_model()
