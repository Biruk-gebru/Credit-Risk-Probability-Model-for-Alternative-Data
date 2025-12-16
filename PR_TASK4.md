# Task 4: Proxy Target Variable Engineering

## 🚀 Overview
This PR implements the logic for creating a proxy target variable for credit risk modeling. Since the dataset lacks a direct "default" label, we use **RFM Analysis** combined with **K-Means Clustering** to identify high-risk customers.

## 🛠 Key Changes
- **RFM Analysis**: Implemented calculation of Recency, Frequency, and Monetary metrics for each customer.
- **Clustering**: Applied K-Means clustering (k=3) to segment customers based on their RFM behavior.
- **Risk Labeling**: Defined a heuristic to identify the "High-Risk" cluster (High Recency, Low Frequency/Monetary) and assigned a binary target variable `is_high_risk`.
- **Integration**: Merged the new target variable back into the main dataset.

## 📂 Files Added
- `src/proxy_target.py`: Class containing the logic for RFM, clustering, and labeling.
- `src/scripts/run_task4.py`: Script to execute the engineering process and save the labeled dataset.

## 📊 Results
- The processed dataset now includes an `is_high_risk` column, ready for model training in Task 5.
