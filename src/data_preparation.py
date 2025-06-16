import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv('data/MachineLearningRating_v3.txt', sep='|', low_memory=False)

# Subset for claim severity (TotalClaims > 0)
df_severity = df[df['TotalClaims'] > 0].copy()

# Handle missing data
df_severity['Province'] = df_severity['Province'].fillna('Unknown')
df_severity['Gender'] = df_severity['Gender'].replace(['Not specified', ''], np.nan).fillna('Unknown')
df_severity['VehicleType'] = df_severity['VehicleType'].fillna('Unknown')
df_severity['PostalCode'] = df_severity['PostalCode'].astype(str).str.strip()
df_severity['make'] = df_severity['make'].fillna('Unknown')
df_severity = df_severity.dropna(subset=['TotalPremium', 'TotalClaims'])

# Feature engineering for severity
df_severity['ClaimFrequency'] = df_severity.groupby('PolicyID')['TotalClaims'].transform('count')
df_severity['PolicyAge'] = (pd.to_datetime(df_severity['TransactionMonth']).max() - 
                            pd.to_datetime(df_severity['TransactionMonth'])).dt.days / 365
df_severity['PremiumPerClaim'] = df_severity['TotalPremium'] / (df_severity['TotalClaims'] + 1e-5)
df_severity['IsHighRiskProvince'] = df_severity['Province'].isin(['Free State', 'KwaZulu-Natal']).astype(int)

# Encode categorical variables
categorical_cols = ['Province', 'Gender', 'VehicleType', 'make']
for col in categorical_cols:
    le = LabelEncoder()
    df_severity[f'{col}_encoded'] = le.fit_transform(df_severity[col])

# Features and target for severity
features_severity = ['TotalPremium', 'PolicyAge', 'PremiumPerClaim', 'IsHighRiskProvince', 
                     'Province_encoded', 'Gender_encoded', 'VehicleType_encoded', 'make_encoded']
X_severity = df_severity[features_severity]
y_severity = df_severity['TotalClaims']

# Train-test split (80:20)
X_train_sev, X_test_sev, y_train_sev, y_test_sev = train_test_split(
    X_severity, y_severity, test_size=0.2, random_state=42
)

# Save severity data
X_train_sev.to_csv('data/X_train_severity.csv', index=False)
X_test_sev.to_csv('data/X_test_severity.csv', index=False)
y_train_sev.to_csv('data/y_train_severity.csv', index=False)
y_test_sev.to_csv('data/y_test_severity.csv', index=False)

# Binary classification (claim probability)
df_class = df.copy()
df_class['HasClaim'] = df_class['TotalClaims'] > 0
df_class['Province'] = df_class['Province'].fillna('Unknown')
df_class['Gender'] = df_class['Gender'].replace(['Not specified', ''], np.nan).fillna('Unknown')
df_class['VehicleType'] = df_class['VehicleType'].fillna('Unknown')
df_class['make'] = df_class['make'].fillna('Unknown')
df_class['PostalCode'] = df_class['PostalCode'].astype(str).str.strip()
df_class['PolicyAge'] = (pd.to_datetime(df_class['TransactionMonth']).max() - 
                         pd.to_datetime(df_class['TransactionMonth'])).dt.days / 365
df_class['IsHighRiskProvince'] = df_class['Province'].isin(['Free State', 'KwaZulu-Natal']).astype(int)

# Encode categorical variables for classification
for col in categorical_cols:
    le = LabelEncoder()
    df_class[f'{col}_encoded'] = le.fit_transform(df_class[col])

# Features for classification (exclude PremiumPerClaim)
features_class = ['TotalPremium', 'PolicyAge', 'IsHighRiskProvince', 
                  'Province_encoded', 'Gender_encoded', 'VehicleType_encoded', 'make_encoded']
X_class = df_class[features_class]
y_class = df_class['HasClaim']

# Train-test split (80:20, stratified)
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42, stratify=y_class
)

# Save classification data
X_train_class.to_csv('data/X_train_class.csv', index=False)
X_test_class.to_csv('data/X_test_class.csv', index=False)
y_train_class.to_csv('data/y_train_class.csv', index=False)
y_test_class.to_csv('data/y_test_class.csv', index=False)

print("Data preparation complete. Saved to data/ directory.")