import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class ProxyTargetEngineer:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.rfm = None
        self.rfm_scaled = None
        self.kmeans = None

    def calculate_rfm(self, customer_id_col='CustomerId', transaction_time_col='TransactionStartTime', amount_col='Amount'):
        """
        Calculate Recency, Frequency, and Monetary value for each customer.
        """
        # Ensure transaction time is datetime
        self.data[transaction_time_col] = pd.to_datetime(self.data[transaction_time_col])
        
        # Calculate Recency, Frequency, Monetary
        # Recency: Days since last transaction
        # Frequency: Number of transactions
        # Monetary: Total amount spent
        
        last_transaction_date = self.data[transaction_time_col].max()
        
        self.rfm = self.data.groupby(customer_id_col).agg({
            transaction_time_col: lambda x: (last_transaction_date - x.max()).days,
            'TransactionId': 'count',
            amount_col: 'sum'
        }).reset_index()
        
        self.rfm.rename(columns={
            transaction_time_col: 'Recency',
            'TransactionId': 'Frequency',
            amount_col: 'Monetary'
        }, inplace=True)
        
        return self.rfm

    def perform_clustering(self, n_clusters=3, random_state=42):
        """
        Perform K-Means clustering on RFM data.
        """
        if self.rfm is None:
            raise ValueError("RFM metrics not calculated. Call calculate_rfm() first.")
            
        # Scale the data
        scaler = StandardScaler()
        self.rfm_scaled = scaler.fit_transform(self.rfm[['Recency', 'Frequency', 'Monetary']])
        
        # KMeans
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.rfm['Cluster'] = self.kmeans.fit_predict(self.rfm_scaled)
        
        return self.rfm

    def assign_risk_label(self):
        """
        Assign 'High-Risk' label based on clusters.
        We assume the cluster with the lowest Monetary and Frequency, and highest Recency is high risk.
        Or we can define high risk as low RFM score.
        For this task, we'll analyze the clusters to determine which one is 'high risk'.
        
        Strategy:
        - Calculate mean R, F, M for each cluster.
        - High Risk: High Recency (haven't seen them in a while), Low Frequency, Low Monetary.
        - Low Risk: Low Recency, High Frequency, High Monetary.
        """
        if 'Cluster' not in self.rfm.columns:
            raise ValueError("Clustering not performed. Call perform_clustering() first.")
            
        cluster_summary = self.rfm.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean'
        })
        
        # Identify High Risk Cluster
        # We look for the cluster with highest Recency and lowest Frequency/Monetary
        # A simple heuristic: Normalize the means and find the "worst" cluster.
        # Or just pick the one with highest Recency as a proxy for churn/risk.
        
        # Let's define a risk score for each cluster: R_norm - F_norm - M_norm
        # Higher score -> Higher Risk
        
        scaler = StandardScaler()
        summary_scaled = scaler.fit_transform(cluster_summary)
        summary_df = pd.DataFrame(summary_scaled, columns=['Recency', 'Frequency', 'Monetary'], index=cluster_summary.index)
        
        summary_df['RiskScore'] = summary_df['Recency'] - summary_df['Frequency'] - summary_df['Monetary']
        
        high_risk_cluster = summary_df['RiskScore'].idxmax()
        
        # Assign label
        self.rfm['is_high_risk'] = self.rfm['Cluster'].apply(lambda x: 1 if x == high_risk_cluster else 0)
        
        return self.rfm

    def integrate_target(self):
        """
        Merge the risk label back into the main dataset.
        """
        if 'is_high_risk' not in self.rfm.columns:
            raise ValueError("Risk label not assigned. Call assign_risk_label() first.")
            
        # Merge only the is_high_risk column
        data_labeled = self.data.merge(self.rfm[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')
        
        return data_labeled
