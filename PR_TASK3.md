# Feature Engineering Implementation (Task 3)

## Description
This PR implements the feature engineering pipeline as part of Task 3.

## Changes
- **Created `src/data_processing.py`** with reusable transformer classes:
  - `DateFeatureExtractor`: Extracts temporal features (hour, day, month, year).
  - `CustomerAggregator`: Aggregates transaction metrics per customer.
  - `CategoricalEncoder`: Supports One-Hot and Label encoding.
  - `MissingValueImputer`: Handles missing data.
  - `WoEEncoder` & `NumericalScaler`: For advanced feature transformation.
- **Added `notebooks/02_feature_engineering.ipynb`** to demonstrate and validate the feature engineering steps.
