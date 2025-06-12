import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('data/MachineLearningRating_v3.txt', sep='|', low_memory=False)

# Clean data
df['PostalCode'] = df['PostalCode'].astype(str).str.strip()
df['Gender'] = df['Gender'].str.strip().replace({'Not specified': np.nan, '': np.nan})
df['VehicleType'] = df['VehicleType'].str.strip().fillna('Unknown')
df['Province'] = df['Province'].str.strip().fillna('Unknown')

# Metrics
df['HasClaim'] = df['TotalClaims'] > 0
df['Margin'] = df['TotalPremium'] - df['TotalClaims']
print(f"Total rows: {len(df)}, Total claims: {df['HasClaim'].sum()}")

# Hypothesis 1: Province
print("\nH0: No risk differences across provinces")
contingency_province = pd.crosstab(df['Province'], df['HasClaim'])
chi2_prov, p_prov, _, _ = chi2_contingency(contingency_province)
severity_prov = df[df['HasClaim']].groupby('Province')['TotalClaims'].mean()
print(f"Claim Frequency (Chi-squared): p-value = {p_prov:.4f}")
print(f"Claim Severity by Province:\n{severity_prov}")

# Hypothesis 2: Zip Code
print("\nH0: No risk differences between zip codes")
zip_claims = df[df['HasClaim']].groupby('PostalCode').size()
valid_zips = zip_claims[zip_claims >= 20].index  # Require 20+ claims
if len(valid_zips) < 2:
    print("Error: Fewer than 2 zip codes with 20+ claims")
    p_freq, p_sev = np.nan, np.nan
else:
    top_zips = df[df['PostalCode'].isin(valid_zips)]['PostalCode'].value_counts().head(2).index
    group_a = df[df['PostalCode'] == top_zips[0]]
    group_b = df[df['PostalCode'] == top_zips[1]]
    print(f"Comparing PostalCodes: {top_zips[0]} vs {top_zips[1]}")
    print(f"Group A size: {len(group_a)}, Claims: {group_a['HasClaim'].sum()}")
    print(f"Group B size: {len(group_b)}, Claims: {group_b['HasClaim'].sum()}")
    print("Checking group similarity (VehicleType):")
    vehicle_crosstab = pd.crosstab(df['PostalCode'].isin([top_zips[0]]), df['VehicleType'])
    print(vehicle_crosstab)
    contingency_zip = pd.crosstab(df['PostalCode'].isin(top_zips), df['HasClaim'])
    chi2_zip, p_freq, _, _ = chi2_contingency(contingency_zip)
    sev_a = group_a[group_a['HasClaim']]['TotalClaims']
    sev_b = group_b[group_b['HasClaim']]['TotalClaims']
    t_stat_sev, p_sev = ttest_ind(sev_a, sev_b, equal_var=False)
    print(f"Claim Frequency (Chi-squared): p-value = {p_freq:.4f}")
    print(f"Claim Severity (t-test): p-value = {p_sev:.4f}")

# Hypothesis 3: Margin
print("\nH0: No significant margin difference between zip codes")
if 'group_a' in locals() and len(group_a) > 0 and len(group_b) > 0:
    t_stat_margin, p_margin = ttest_ind(group_a['Margin'], group_b['Margin'], equal_var=False)
    print(f"Margin (t-test): p-value = {p_margin:.4f}")
else:
    p_margin = np.nan

# Hypothesis 4: Gender
print("\nH0: No significant risk difference between Women and Men")
df_gender = df[df['Gender'].isin(['Male', 'Female'])]
male = df_gender[df_gender['Gender'] == 'Male']
female = df_gender[df_gender['Gender'] == 'Female']
print(f"Male size: {len(male)}, Claims: {male['HasClaim'].sum()}")
print(f"Female size: {len(female)}, Claims: {female['HasClaim'].sum()}")
print("Checking gender similarity (Province):")
gender_crosstab = pd.crosstab(df['Gender'].isin(['Male']), df['Province'])
print(gender_crosstab)
if male['HasClaim'].sum() >= 50 and female['HasClaim'].sum() >= 50:
    contingency_gender = pd.crosstab(df['Gender'].isin(['Male', 'Female']), df['HasClaim'])
    chi2_gender, p_gender_freq, _, _ = chi2_contingency(contingency_gender)
    print(f"Claim Frequency (Chi-squared): p-value = {p_gender_freq:.4f}")
else:
    print("Insufficient claims data for gender frequency test")
    p_gender_freq = np.nan
if male['HasClaim'].sum() >= 10 and female['HasClaim'].sum() >= 10:
    sev_male = male[male['HasClaim']]['TotalClaims']
    sev_female = female[female['HasClaim']]['TotalClaims']
    t_stat_sev, p_gender_sev = ttest_ind(sev_male, sev_female, equal_var=False)
    print(f"Claim Severity (t-test): p-value = {p_gender_sev:.4f}")
else:
    print("Insufficient claims data for gender severity test")
    p_gender_sev = np.nan

# Save results
with open('reports/ab_testing_results.txt', 'w') as f:
    f.write("A/B Testing Results\n")
    f.write(f"Province Risk: p-value = {p_prov:.4f}\n")
    f.write(f"Zip Code Frequency: p-value = {p_freq:.4f}\n")
    f.write(f"Zip Code Severity: p-value = {p_sev:.4f}\n")
    f.write(f"Zip Code Margin: p-value = {p_margin:.4f}\n")
    f.write(f"Gender Frequency: p-value = {p_gender_freq:.4f}\n")
    f.write(f"Gender Severity: p-value = {p_gender_sev:.4f}\n")