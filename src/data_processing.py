import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts date-related features from a timestamp column.
    """
    def __init__(self, date_col='TransactionStartTime'):
        self.date_col = date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.date_col in X.columns:
            X[self.date_col] = pd.to_datetime(X[self.date_col])
            X['TransactionHour'] = X[self.date_col].dt.hour
            X['TransactionDay'] = X[self.date_col].dt.day
            X['TransactionMonth'] = X[self.date_col].dt.month
            X['TransactionYear'] = X[self.date_col].dt.year
        return X

class CustomerAggregator(BaseEstimator, TransformerMixin):
    """
    Aggregates transaction data to customer level.
    """
    def __init__(self, customer_col='CustomerId', agg_cols=None, agg_funcs=None):
        self.customer_col = customer_col
        self.agg_cols = agg_cols if agg_cols else ['Amount', 'Value']
        self.agg_funcs = agg_funcs if agg_funcs else ['sum', 'mean', 'count', 'std']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # Ensure columns exist
        valid_cols = [col for col in self.agg_cols if col in X.columns]
        
        if not valid_cols:
            return pd.DataFrame() # Return empty if no columns to aggregate

        agg_dict = {col: self.agg_funcs for col in valid_cols}
        
        # Group by customer and aggregate
        X_agg = X.groupby(self.customer_col).agg(agg_dict)
        
        # Flatten MultiIndex columns
        X_agg.columns = [f"{col}_{func}" for col, func in X_agg.columns.values]
        X_agg.reset_index(inplace=True)
        
        return X_agg

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes categorical features using One-Hot Encoding or Label Encoding.
    """
    def __init__(self, method='onehot', columns=None):
        self.method = method
        self.columns = columns
        self.encoder = None

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if self.method == 'onehot':
            self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            self.encoder.fit(X[self.columns])
        elif self.method == 'label':
            # LabelEncoder is usually for 1D array, so we might need a dict of encoders
            self.encoder = {col: LabelEncoder().fit(X[col].astype(str)) for col in self.columns}
        return self

    def transform(self, X):
        X = X.copy()
        if self.method == 'onehot':
            encoded_cols = self.encoder.get_feature_names_out(self.columns)
            encoded_data = self.encoder.transform(X[self.columns])
            df_encoded = pd.DataFrame(encoded_data, columns=encoded_cols, index=X.index)
            X = pd.concat([X.drop(columns=self.columns), df_encoded], axis=1)
        elif self.method == 'label':
            for col, le in self.encoder.items():
                # Handle unseen labels by assigning a default or keeping as is (LabelEncoder is strict)
                # For simplicity, we'll convert to string and transform, handling errors?
                # A robust way is to use map and fillna
                X[col] = X[col].astype(str).map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
        return X

class MissingValueImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values.
    """
    def __init__(self, strategy='mean', columns=None):
        self.strategy = strategy
        self.columns = columns
        self.imputer = None

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.select_dtypes(include=['number']).columns.tolist()
        
        self.imputer = SimpleImputer(strategy=self.strategy)
        self.imputer.fit(X[self.columns])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.columns] = self.imputer.transform(X[self.columns])
        return X

class WoEEncoder(BaseEstimator, TransformerMixin):
    """
    Weight of Evidence (WoE) Encoder and Information Value (IV) Calculator.
    """
    def __init__(self, target_col='FraudResult', bins=10):
        self.target_col = target_col
        self.bins = bins
        self.woe_dict = {}
        self.iv_dict = {}

    def fit(self, X, y=None):
        # Assuming X contains the target column if y is None, or we use y
        if y is None:
            if self.target_col in X.columns:
                y = X[self.target_col]
                X_features = X.drop(columns=[self.target_col])
            else:
                raise ValueError("Target column not found in X and y not provided.")
        else:
            X_features = X.copy()

        # Calculate WoE for each column
        for col in X_features.columns:
            if pd.api.types.is_numeric_dtype(X_features[col]):
                # Bin numerical features
                try:
                    X_features[col] = pd.qcut(X_features[col], self.bins, duplicates='drop')
                except Exception:
                     # Fallback if qcut fails (e.g. too many duplicates)
                     pass
            
            # Calculate WoE
            df = pd.DataFrame({'feature': X_features[col], 'target': y})
            
            # Total positive and negative
            total_pos = df['target'].sum()
            total_neg = df['target'].count() - total_pos
            
            if total_pos == 0 or total_neg == 0:
                continue # Skip if target is constant

            # Group by feature
            grouped = df.groupby('feature')['target'].agg(['count', 'sum'])
            grouped['non_event'] = grouped['count'] - grouped['sum']
            grouped['event'] = grouped['sum']
            
            # Avoid division by zero
            grouped['event_rate'] = (grouped['event'] + 0.5) / total_pos
            grouped['non_event_rate'] = (grouped['non_event'] + 0.5) / total_neg
            
            grouped['woe'] = np.log(grouped['event_rate'] / grouped['non_event_rate'])
            grouped['iv'] = (grouped['event_rate'] - grouped['non_event_rate']) * grouped['woe']
            
            self.woe_dict[col] = grouped['woe'].to_dict()
            self.iv_dict[col] = grouped['iv'].sum()
            
        return self

    def transform(self, X):
        X = X.copy()
        # Replace values with WoE
        for col, mapping in self.woe_dict.items():
            if col in X.columns:
                # If numerical, we need to bin again using the same bins?
                # This implementation is simplified and assumes categorical or already binned for transform
                # For a robust implementation, we'd need to store the bin edges.
                # For now, we'll map exact matches (categorical).
                # If numerical, this simple map won't work perfectly without binning logic in transform.
                
                # Simplified: only map if it matches the keys (categorical)
                if not pd.api.types.is_numeric_dtype(X[col]):
                     X[col] = X[col].map(mapping).fillna(0) # Fill unknown with 0 (neutral WoE)
        return X

class NumericalScaler(BaseEstimator, TransformerMixin):
    """
    Scales numerical features using StandardScaler or MinMaxScaler.
    """
    def __init__(self, columns=None, method='standard'):
        self.columns = columns
        self.method = method
        self.scaler = None

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.select_dtypes(include=['number']).columns.tolist()
        
        if self.method == 'standard':
            self.scaler = StandardScaler()
        elif self.method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
        
        self.scaler.fit(X[self.columns])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.columns] = self.scaler.transform(X[self.columns])
        return X

def calculate_iv(df, target_col):
    """
    Helper function to calculate IV for all columns.
    """
    encoder = WoEEncoder(target_col=target_col)
    encoder.fit(df)
    return pd.Series(encoder.iv_dict).sort_values(ascending=False)
