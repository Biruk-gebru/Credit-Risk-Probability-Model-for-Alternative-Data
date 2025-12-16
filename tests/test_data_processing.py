import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import (
    DateFeatureExtractor, 
    CustomerAggregator, 
    CategoricalEncoder, 
    NumericalScaler,
    MissingValueImputer
)

@pytest.fixture
def sample_data():
    data = {
        'TransactionStartTime': ['2023-01-01 10:00:00', '2023-01-01 11:00:00', '2023-01-02 10:00:00'],
        'CustomerId': ['C1', 'C1', 'C2'],
        'Amount': [100.0, 200.0, 300.0],
        'Category': ['A', 'B', 'A'],
        'Value': [10, 20, 30]
    }
    return pd.DataFrame(data)

def test_date_feature_extractor(sample_data):
    extractor = DateFeatureExtractor(date_col='TransactionStartTime')
    df_transformed = extractor.fit_transform(sample_data)
    
    assert 'TransactionHour' in df_transformed.columns
    assert 'TransactionDay' in df_transformed.columns
    assert 'TransactionMonth' in df_transformed.columns
    assert 'TransactionYear' in df_transformed.columns
    assert df_transformed['TransactionHour'].iloc[0] == 10

def test_customer_aggregator(sample_data):
    aggregator = CustomerAggregator(customer_col='CustomerId', agg_cols=['Amount'], agg_funcs=['sum', 'mean'])
    df_transformed = aggregator.fit_transform(sample_data)
    
    assert 'Amount_sum' in df_transformed.columns
    assert 'Amount_mean' in df_transformed.columns
    
    # Check values for C1 (first row)
    # C1 has amounts 100 and 200. Sum = 300. Mean = 150.
    assert df_transformed[df_transformed['CustomerId'] == 'C1']['Amount_sum'].iloc[0] == 300.0
    assert df_transformed[df_transformed['CustomerId'] == 'C1']['Amount_mean'].iloc[0] == 150.0

def test_categorical_encoder(sample_data):
    encoder = CategoricalEncoder(method='onehot', columns=['Category'])
    df_transformed = encoder.fit_transform(sample_data)
    
    # Check if encoded columns exist
    # OneHotEncoder feature names depend on implementation. Usually Category_A, Category_B
    # Let's check if any column starts with Category_
    encoded_cols = [c for c in df_transformed.columns if c.startswith('Category_')]
    assert len(encoded_cols) >= 2
    assert 'Category' not in df_transformed.columns

def test_numerical_scaler(sample_data):
    scaler = NumericalScaler(columns=['Amount'], method='standard')
    df_transformed = scaler.fit_transform(sample_data)
    
    # Check if mean is approx 0
    assert df_transformed['Amount'].mean() == pytest.approx(0, abs=1e-6)

def test_missing_value_imputer():
    df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 4.0]})
    imputer = MissingValueImputer(strategy='mean', columns=['A'])
    df_transformed = imputer.fit_transform(df)
    
    assert df_transformed['A'].isnull().sum() == 0
    assert df_transformed['A'].iloc[2] == pytest.approx(2.333, abs=0.01)
