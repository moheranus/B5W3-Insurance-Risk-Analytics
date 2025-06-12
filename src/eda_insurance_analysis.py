
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import logging

# Set up logging
logging.basicConfig(filename='eda_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info("Starting EDA script")

# Set up plotting style
plt.style.use('ggplot')
sns.set_palette("husl")

# Create visualizations directory
os.makedirs('visualizations', exist_ok=True)

# Load the dataset
try:
    logging.info("Loading dataset...")
    df = pd.read_csv('data/MachineLearningRating_v3.txt', sep='|', low_memory=False)
    logging.info("Dataset loaded successfully.")
except FileNotFoundError:
    logging.error("Error: 'data/MachineLearningRating_v3.txt' not found.")
    print("Error: 'data/MachineLearningRating_v3.txt' not found.")
    exit(1)

# Sample dataset for visualizations
SAMPLE_SIZE = 10000
if len(df) > SAMPLE_SIZE:
    df_sample = df.sample(n=SAMPLE_SIZE, random_state=42)
else:
    df_sample = df.copy()

# Data Summarization: Descriptive Statistics
def data_summarization(df):
    logging.info("Running data summarization...")
    numerical_cols = [col for col in ['TotalPremium', 'TotalClaims', 'CustomValueEstimate', 'SumInsured', 'CalculatedPremiumPerTerm'] if col in df.columns]
    if numerical_cols:
        desc_stats = df[numerical_cols].describe()
        print("Descriptive Statistics for Numerical Features:")
        print(desc_stats)
        print("\nVariability (Standard Deviation):")
        print(df[numerical_cols].std())
    else:
        print("No numerical columns found.")
    print("\nData Types and Structure:")
    print(df.dtypes)
    logging.info("Data summarization completed.")

# Data Quality Assessment
def data_quality_assessment(df):
    logging.info("Running data quality assessment...")
    print("\nMissing Values:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    print("\nPercentage of Missing Values:")
    missing_percent = (df.isnull().sum() / len(df)) * 100
    print(missing_percent[missing_percent > 0])
    logging.info("Data quality assessment completed.")

# Univariate Analysis
def univariate_analysis(df):
    logging.info("Running univariate analysis...")
    try:
        numerical_cols = [col for col in ['TotalPremium', 'TotalClaims', 'CustomValueEstimate'] if col in df.columns]
        if numerical_cols:
            plt.figure(figsize=(15, 5))
            for i, col in enumerate(numerical_cols, 1):
                plt.subplot(1, len(numerical_cols), i)
                sns.histplot(df[col].dropna(), kde=True)
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig('visualizations/numerical_distributions.png')
            plt.close()
        
        categorical_cols = [col for col in ['Province', 'VehicleType', 'Gender'] if col in df.columns]
        if categorical_cols:
            plt.figure(figsize=(15, 5))
            for i, col in enumerate(categorical_cols, 1):
                plt.subplot(1, len(categorical_cols), i)
                df[col].value_counts().plot(kind='bar')
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('visualizations/categorical_distributions.png')
            plt.close()
        logging.info("Univariate analysis completed.")
    except Exception as e:
        logging.error(f"Error in univariate analysis: {e}")
        print(f"Error in univariate analysis: {e}")

# Bivariate/Multivariate Analysis
def bivariate_analysis(df):
    logging.info("Running bivariate analysis...")
    try:
        if 'TotalPremium' in df.columns and 'TotalClaims' in df.columns:
            df['LossRatio'] = df['TotalClaims'] / df['TotalPremium'].replace(0, np.nan)
            
            if 'Province' in df.columns:
                plt.figure(figsize=(10, 6))
                sns.barplot(x='Province', y='LossRatio', data=df)
                plt.title('Loss Ratio by Province')
                plt.xticks(rotation=45)
                plt.savefig('visualizations/loss_ratio_province.png')
                plt.close()
            
            if 'VehicleType' in df.columns:
                plt.figure(figsize=(10, 6))
                sns.barplot(x='VehicleType', y='LossRatio', data=df)
                plt.title('Loss Ratio by Vehicle Type')
                plt.xticks(rotation=45)
                plt.savefig('visualizations/loss_ratio_vehicle_type.png')
                plt.close()
            
            if 'Gender' in df.columns:
                plt.figure(figsize=(10, 6))
                sns.barplot(x='Gender', y='LossRatio', data=df)
                plt.title('Loss Ratio by Gender')
                plt.savefig('visualizations/loss_ratio_gender.png')
                plt.close()
            
            numerical_cols = [col for col in ['TotalPremium', 'TotalClaims', 'CustomValueEstimate'] if col in df.columns]
            if numerical_cols:
                corr_matrix = df[numerical_cols].corr()
                plt.figure(figsize=(8, 6))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
                plt.title('Correlation Matrix of Numerical Features')
                plt.savefig('visualizations/correlation_matrix.png')
                plt.close()
            
            if 'PostalCode' in df.columns:
                plt.figure(figsize=(10, 6))
                sns.scatterplot(x='TotalPremium', y='TotalClaims', hue='PostalCode', size='PostalCode', data=df)
                plt.title('TotalPremium vs TotalClaims by PostalCode')
                plt.savefig('visualizations/premium_vs_claims_postalcode.png')
                plt.close()
        logging.info("Bivariate analysis completed.")
    except Exception as e:
        logging.error(f"Error in bivariate analysis: {e}")
        print(f"Error in bivariate analysis: {e}")

# Temporal Trends Analysis
def temporal_trends(df):
    logging.info("Running temporal trends analysis...")
    try:
        if 'TransactionMonth' in df.columns:
            df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
            monthly_trends = df.groupby(df['TransactionMonth'].dt.to_period('M')).agg({
                'TotalClaims': ['mean', 'count'],
                'TotalPremium': 'mean'
            })
            monthly_trends.columns = ['Avg_Claims', 'Claim_Count', 'Avg_Premium']
            plt.figure(figsize=(12, 6))
            plt.plot(monthly_trends.index.to_timestamp(), monthly_trends['Avg_Claims'], label='Average Claims')
            plt.plot(monthly_trends.index.to_timestamp(), monthly_trends['Avg_Premium'], label='Average Premium')
            plt.title('Monthly Trends in Claims and Premiums')
            plt.xlabel('Month')
            plt.ylabel('Amount')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('visualizations/monthly_trends.png')
            plt.close()
        logging.info("Temporal trends analysis completed.")
    except Exception as e:
        logging.error(f"Error in temporal trends: {e}")
        print(f"Error in temporal trends: {e}")

# Outlier Detection
def outlier_detection(df):
    logging.info("Running outlier detection...")
    try:
        numerical_cols = [col for col in ['TotalPremium', 'TotalClaims', 'CustomValueEstimate'] if col in df.columns]
        if numerical_cols:
            plt.figure(figsize=(15, 5))
            for i, col in enumerate(numerical_cols, 1):
                plt.subplot(1, len(numerical_cols), i)
                sns.boxplot(y=df[col])
                plt.title(f'Box Plot of {col}')
            plt.tight_layout()
            plt.savefig('visualizations/box_plots.png')
            plt.close()
        logging.info("Outlier detection completed.")
    except Exception as e:
        logging.error(f"Error in outlier detection: {e}")
        print(f"Error in outlier detection: {e}")

# Vehicle Make/Model Analysis
def vehicle_analysis(df):
    logging.info("Running vehicle analysis...")
    try:
        if 'make' in df.columns:
            make_claims = df.groupby('make')['TotalClaims'].agg(['mean', 'count']).sort_values(by='mean')
            plt.figure(figsize=(12, 6))
            sns.barplot(x=make_claims.index[-10:], y=make_claims['mean'][-10:])
            plt.title('Top 10 Vehicle Makes by Average Claim Amount')
            plt.xticks(rotation=45)
            plt.ylabel('Average Claim Amount')
            plt.savefig('visualizations/top_makes_claims.png')
            plt.close()
        logging.info("Vehicle analysis completed.")
    except Exception as e:
        logging.error(f"Error in vehicle analysis: {e}")
        print(f"Error in vehicle analysis: {e}")

# Main execution
def main():
    logging.info("Starting main EDA execution")
    data_summarization(df)
    data_quality_assessment(df)
    univariate_analysis(df_sample)
    bivariate_analysis(df_sample)
    temporal_trends(df)
    outlier_detection(df_sample)
    vehicle_analysis(df_sample)
    print("EDA completed. Visualizations saved in visualizations/ folder.")
    logging.info("EDA completed successfully")

if __name__ == "__main__":
    main()
