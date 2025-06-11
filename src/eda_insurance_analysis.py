import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Set up plotting style
plt.style.use('seaborn')
sns.set_palette("husl")

# Load the dataset (assuming it's in data/ folder)
df = pd.read_csv('data/insurance_data.csv')

# Data Summarization: Descriptive Statistics
def data_summarization(df):
    print("Descriptive Statistics for Numerical Features:")
    numerical_cols = ['TotalPremium', 'TotalClaims', 'CustomValueEstimate', 'SumInsured', 'CalculatedPremiumPerTerm']
    desc_stats = df[numerical_cols].describe()
    print(desc_stats)
    print("\nVariability (Standard Deviation):")
    print(df[numerical_cols].std())
    print("\nData Types and Structure:")
    print(df.dtypes)

# Data Quality Assessment
def data_quality_assessment(df):
    print("\nMissing Values:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    print("\nPercentage of Missing Values:")
    missing_percent = (df.isnull().sum() / len(df)) * 100
    print(missing_percent[missing_percent > 0])

# Univariate Analysis
def univariate_analysis(df):
    numerical_cols = ['TotalPremium', 'TotalClaims', 'CustomValueEstimate']
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(1, 3, i)
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('visualizations/numerical_distributions.png')
    plt.close()
    categorical_cols = ['Province', 'VehicleType', 'Gender']
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(categorical_cols, 1):
        plt.subplot(1, 3, i)
        df[col].value_counts().plot(kind='bar')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/categorical_distributions.png')
    plt.close()

# Bivariate/Multivariate Analysis
def bivariate_analysis(df):
    df['LossRatio'] = df['TotalClaims'] / df['TotalPremium'].replace(0, np.nan)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Province', y='LossRatio', data=df)
    plt.title('Loss Ratio by Province')
    plt.xticks(rotation=45)
    plt.savefig('visualizations/loss_ratio_province.png')
    plt.close()
    numerical_cols = ['TotalPremium', 'TotalClaims', 'CustomValueEstimate']
    corr_matrix = df[numerical_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Numerical Features')
    plt.savefig('visualizations/correlation_matrix.png')
    plt.close()
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='TotalPremium', y='TotalClaims', hue='PostalCode', size='PostalCode', data=df.sample(1000))
    plt.title('TotalPremium vs TotalClaims by ZipCode')
    plt.savefig('visualizations/premium_vs_claims_zipcode.png')
    plt.close()

# Temporal Trends Analysis
def temporal_trends(df):
    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
    monthly_trends = df.groupby(df['TransactionMonth'].dt.to_period('M')).agg({
        'TotalClaims': ['mean', 'count'],
        'TotalPremium': 'mean'
    })
    monthly_trends.columns = ['Avg_Claims', 'Claim_Count', 'Avg_Premium']
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_trends.index.astype(str), monthly_trends['Avg_Claims'], label='Average Claims')
    plt.plot(monthly_trends.index.astype(str), monthly_trends['Avg_Premium'], label='Average Premium')
    plt.title('Monthly Trends in Claims and Premiums')
    plt.xlabel('Month')
    plt.ylabel('Amount')
    plt.legend()
    plt.xticks(rotation=45)
    plt.savefig('visualizations/monthly_trends.png')
    plt.close()

# Outlier Detection
def outlier_detection(df):
    numerical_cols = ['TotalPremium', 'TotalClaims', 'CustomValueEstimate']
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(1, 3, i)
        sns.boxplot(y=df[col])
        plt.title(f'Box Plot of {col}')
    plt.tight_layout()
    plt.savefig('visualizations/box_plots.png')
    plt.close()

# Vehicle Make/Model Analysis
def vehicle_analysis(df):
    make_claims = df.groupby('Make')['TotalClaims'].agg(['mean', 'count']).sort_values(by='mean')
    plt.figure(figsize=(12, 6))
    sns.barplot(x=make_claims.index[-10:], y=make_claims['mean'][-10:])
    plt.title('Top 10 Vehicle Makes by Average Claim Amount')
    plt.xticks(rotation=45)
    plt.ylabel('Average Claim Amount')
    plt.savefig('visualizations/top_makes_claims.png')
    plt.close()

# Main execution
def main():
    print("Starting EDA for Insurance Dataset")
    data_summarization(df)
    data_quality_assessment(df)
    univariate_analysis(df)
    bivariate_analysis(df)
    temporal_trends(df)
    outlier_detection(df)
    vehicle_analysis(df)
    print("EDA completed. Visualizations saved in visualizations/ folder.")

if __name__ == "__main__":
    main()