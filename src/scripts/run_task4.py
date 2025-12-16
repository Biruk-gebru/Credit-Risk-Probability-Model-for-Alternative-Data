import os
import pandas as pd
import sys

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.proxy_target import ProxyTargetEngineer

def main():
    # Paths
    input_path = 'data/processed/data_featured.csv'
    output_path = 'data/processed/data_labeled.csv'
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        return

    # Load data
    print("Loading data...")
    df = pd.read_csv(input_path)
    
    # Initialize Engineer
    engineer = ProxyTargetEngineer(df)
    
    # Calculate RFM
    print("Calculating RFM metrics...")
    rfm = engineer.calculate_rfm()
    print(rfm.head())
    
    # Clustering
    print("Performing Clustering...")
    engineer.perform_clustering(n_clusters=3)
    
    # Assign Risk Label
    print("Assigning Risk Labels...")
    engineer.assign_risk_label()
    
    # Integrate
    print("Integrating Target Variable...")
    final_df = engineer.integrate_target()
    
    # Save
    print(f"Saving labeled data to {output_path}...")
    final_df.to_csv(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
