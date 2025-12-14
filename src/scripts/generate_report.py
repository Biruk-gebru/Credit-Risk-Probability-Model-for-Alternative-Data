import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import load_data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import re

# Set style
sns.set_theme(style="whitegrid")

# Paths
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw/data.csv"))
README_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../README.md"))
OUTPUT_DIR = "/home/karanos/kiam/week4/docs"
PLOTS_DIR = "temp_plots"

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Credit Risk Analysis - Interim Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 7)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        self.cell(0, 10, 'Author: Biruk Gebru Jember', 0, 0, 'R')

def get_task1_content():
    try:
        with open(README_PATH, 'r') as f:
            content = f.read()
        
        # Extract content starting from "Credit Scoring Business Understanding"
        match = re.search(r'## Credit Scoring Business Understanding(.*?)$', content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return "Task 1 content not found in README."
    except Exception as e:
        return f"Error reading README: {str(e)}"

def clean_markdown(text):
    # Remove markdown specific characters for better PDF rendering
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
    text = re.sub(r'### ', '', text)  # Headers
    text = re.sub(r'- ', '- ', text)  # Bullets
    return text

def generate_plots(df):
    # 1. Transaction Count by Category
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, y='ProductCategory', order=df['ProductCategory'].value_counts().index)
    plt.title('Transaction Count by Product Category')
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/category_count.png")
    plt.close()

    # 2. Fraud Rate by Category
    fraud_rate = df.groupby('ProductCategory')['FraudResult'].mean().sort_values(ascending=False) * 100
    plt.figure(figsize=(10, 6))
    sns.barplot(x=fraud_rate.values, y=fraud_rate.index)
    plt.title('Fraud Rate by Product Category (%)')
    plt.xlabel('Fraud Rate (%)')
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/fraud_rate.png")
    plt.close()

    # 3. Transaction Value Distribution (Log Scale)
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Amount', bins=50, log_scale=True)
    plt.title('Distribution of Transaction Amounts (Log Scale)')
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/amount_dist.png")
    plt.close()

def create_report():
    pdf = PDF()
    
    # --- Task 1 Section ---
    pdf.add_page()
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 10, '1. Business Understanding (Task 1)', 0, 1, 'L')
    
    task1_text = get_task1_content()
    # Clean and split text to handle it better
    cleaned_text = clean_markdown(task1_text)
    
    pdf.set_font('Arial', '', 9)
    # Write text line by line to avoid issues with very long strings
    for line in cleaned_text.split('\n'):
        if line.strip():
            # Ensure we only use latin-1 compatible characters
            safe_line = line.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 5, safe_line)
            pdf.ln(1)
        else:
            pdf.ln(2)

    # --- Task 2 Section ---
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 10, '2. Exploratory Data Analysis (Task 2)', 0, 1, 'L')
    pdf.set_font('Arial', '', 9)
    pdf.multi_cell(0, 7, 
        "This section presents findings from the analysis of transaction data, "
        "focusing on customer behavior, fraud patterns, and value distributions."
    )
    pdf.ln(5)

    # Dataset Overview
    pdf.set_font('Arial', 'B', 9)
    pdf.cell(0, 8, 'Dataset Structure & Quality:', 0, 1)
    pdf.set_font('Arial', '', 9)
    structure_text = (
        "- Shape: 95,662 rows, 16 columns.\n"
        "- Missing Values: None (0 missing values across all columns).\n"
        "- Data Types: Mix of categorical (ProviderId, ProductId) and numerical (Amount, Value) features.\n"
        "- Outliers: Significant outliers detected in 'Amount' and 'Value' via box plots."
    )
    pdf.multi_cell(0, 7, structure_text)
    pdf.ln(3)

    # Key Statistics
    pdf.set_font('Arial', 'B', 9)
    pdf.cell(0, 8, 'Customer Engagement:', 0, 1)
    pdf.set_font('Arial', '', 9)
    stats_text = (
        "- Average transactions per customer: 25.56\n"
        "- Median transactions per customer: 7\n"
        "- Max transactions by single customer: 4091\n"
        "- Insight: Highly skewed engagement indicating a mix of casual and power users."
    )
    pdf.multi_cell(0, 7, stats_text)
    pdf.ln(3)

    pdf.set_font('Arial', 'B', 9)
    pdf.cell(0, 8, 'Transaction Value Statistics:', 0, 1)
    pdf.set_font('Arial', '', 9)
    value_stats = (
        "- Mean Transaction Value: 9,900.58\n"
        "- Median Transaction Value: 1,000.00\n"
        "- Skewness: 51.29\n"
        "- Insight: Extreme right-skewness requires log-transformation for modeling."
    )
    pdf.multi_cell(0, 7, value_stats)
    pdf.ln(3)

    pdf.set_font('Arial', 'B', 9)
    pdf.cell(0, 8, 'Fraud Analysis:', 0, 1)
    pdf.set_font('Arial', '', 9)
    fraud_stats = (
        "- Overall Fraud Rate: 0.2018%\n"
        "- Highest Risk Category: Transport (8.00%)\n"
        "- Lowest Risk Category: Data Bundles (0.00%)\n"
        "- Insight: Fraud is rare but highly concentrated in specific categories."
    )
    pdf.multi_cell(0, 7, fraud_stats)
    pdf.ln(3)

    # Correlation Analysis (New)
    pdf.set_font('Arial', 'B', 9)
    pdf.cell(0, 8, 'Correlation Analysis:', 0, 1)
    pdf.set_font('Arial', '', 9)
    corr_text = (
        "- Strong positive correlation observed between 'Amount' and 'Value'.\n"
        "- 'FraudResult' shows weak linear correlation with base numerical features, suggesting non-linear patterns."
    )
    pdf.multi_cell(0, 7, corr_text)
    pdf.ln(5)

    # Visualizations
    pdf.add_page()
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 10, '3. Visualizations', 0, 1, 'L')
    
    pdf.image(f"{PLOTS_DIR}/category_count.png", x=10, w=170)
    pdf.ln(5)
    pdf.set_font('Arial', 'I', 8)
    pdf.multi_cell(0, 5, "Figure 1: Transaction volume is dominated by Financial Services and Airtime.")
    pdf.ln(10)

    pdf.image(f"{PLOTS_DIR}/fraud_rate.png", x=10, w=170)
    pdf.ln(5)
    pdf.multi_cell(0, 5, "Figure 2: Despite lower volume, Transport and Utility Bills show disproportionately higher fraud rates.")
    
    pdf.add_page()
    pdf.image(f"{PLOTS_DIR}/amount_dist.png", x=10, w=170)
    pdf.ln(5)
    pdf.multi_cell(0, 5, "Figure 3: Transaction amounts are heavily skewed, necessitating log-scale visualization.")

    # Roadmap Section (New)
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 10, '4. Project Roadmap & Next Steps', 0, 1, 'L')
    
    pdf.set_font('Arial', 'B', 9)
    pdf.cell(0, 8, 'Feature Engineering:', 0, 1)
    pdf.set_font('Arial', '', 9)
    feat_eng_text = (
        "- Aggregate Features: Calculate total transaction sum, count, and average per customer.\n"
        "- Time Features: Extract hour, day, month, and year from transaction timestamps.\n"
        "- Advanced Features: Implement Weight of Evidence (WoE) and Information Value (IV) for risk classification."
    )
    pdf.multi_cell(0, 7, feat_eng_text)
    pdf.ln(3)

    pdf.set_font('Arial', 'B', 9)
    pdf.cell(0, 8, 'Proxy Target Engineering:', 0, 1)
    pdf.set_font('Arial', '', 9)
    target_text = (
        "- RFM Analysis: Calculate Recency, Frequency, and Monetary metrics for each user.\n"
        "- Risk Classification: Use K-Means clustering on RFM scores to assign Good/Bad credit labels (proxy target)."
    )
    pdf.multi_cell(0, 7, target_text)
    pdf.ln(3)

    pdf.set_font('Arial', 'B', 9)
    pdf.cell(0, 8, 'Modeling & Evaluation:', 0, 1)
    pdf.set_font('Arial', '', 9)
    model_text = (
        "- Models: Train Logistic Regression, Random Forest, and GBM models.\n"
        "- Tracking: Use MLflow to track experiments, parameters, and metrics.\n"
        "- Tuning: Perform hyperparameter tuning using Grid Search or Optuna."
    )
    pdf.multi_cell(0, 7, model_text)
    pdf.ln(3)

    pdf.set_font('Arial', 'B', 9)
    pdf.cell(0, 8, 'Deployment (MLOps):', 0, 1)
    pdf.set_font('Arial', '', 9)
    deploy_text = (
        "- API: Serve the best performing model using FastAPI.\n"
        "- Containerization: Dockerize the application for consistent deployment.\n"
        "- CI/CD: Implement a CI/CD pipeline (GitHub Actions) for automated testing and deployment."
    )
    pdf.multi_cell(0, 7, deploy_text)

    # Save
    pdf.output(f"{OUTPUT_DIR}/interim_report.pdf")
    print(f"Report generated successfully at {OUTPUT_DIR}/interim_report.pdf")

if __name__ == "__main__":
    print("Loading data...")
    df = load_data(DATA_PATH)
    print("Generating plots...")
    generate_plots(df)
    print("Creating PDF report...")
    create_report()
    
    # Cleanup
    import shutil
    shutil.rmtree(PLOTS_DIR)
