# Task 2: Exploratory Data Analysis and Project Restructuring

## Description
This PR completes Task 2 (Exploratory Data Analysis) and standardizes the project structure as per the engineering requirements.

## Key Changes
*   **EDA**: Performed comprehensive analysis in `notebooks/eda.ipynb`, uncovering insights on fraud distribution and transaction patterns.
*   **Reporting**: Generated an automated interim PDF report (`docs/interim_report.pdf`) summarizing Business Understanding and EDA findings.
*   **Structure**: Refactored the repository to follow a production-ready layout, including:
    *   `src/` for modular code (data processing, training, API).
    *   `tests/` for unit testing.
    *   `Dockerfile` and `docker-compose.yml` for containerization.
    *   CI/CD workflow (`.github/workflows/ci.yml`).

## Insights
*   **Fraud Distribution**: Fraud rate is low (~0.2%) but highly concentrated in 'Transport' and 'Utility' categories.
*   **Data Skewness**: Transaction values are heavily right-skewed, suggesting the need for log-transformation.
*   **Outliers**: Significant outliers exist in 'Amount' and 'Value' features.

## Next Steps
*   Merge to `main`.
*   Begin Task 3 (Feature Engineering).
