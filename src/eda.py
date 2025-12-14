import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_numerical_distribution(df, column, bins=50):
    """
    Plots the distribution of a numerical column (Histogram and Boxplot).

    Args:
        df (pd.DataFrame): The dataframe.
        column (str): The column name.
        bins (int): Number of bins for histogram.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Histogram
    axes[0].hist(df[column], bins=bins, edgecolor='black', alpha=0.7)
    axes[0].set_title(f'Distribution of {column}', fontsize=14, fontweight='bold')
    axes[0].set_xlabel(column)
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)

    # Box plot
    axes[1].boxplot(df[column], vert=True)
    axes[1].set_title(f'Box Plot of {column}', fontsize=14, fontweight='bold')
    axes[1].set_ylabel(column)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_categorical_distribution(df, column, top_n=10):
    """
    Plots the distribution of a categorical column.

    Args:
        df (pd.DataFrame): The dataframe.
        column (str): The column name.
        top_n (int): Number of top categories to show.
    """
    plt.figure(figsize=(10, 6))
    order = df[column].value_counts().iloc[:top_n].index
    sns.countplot(data=df, x=column, order=order, palette='viridis')
    plt.title(f'Distribution of {column} (Top {top_n})', fontsize=14, fontweight='bold')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.show()

def plot_correlation_matrix(df):
    """
    Plots the correlation matrix of numerical features.

    Args:
        df (pd.DataFrame): The dataframe.
    """
    numerical_cols = df.select_dtypes(include=['number']).columns
    corr_matrix = df[numerical_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix', fontsize=14, fontweight='bold')
    plt.show()
