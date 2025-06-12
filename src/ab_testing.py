import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('data/MachineLearningRating_v3.txt', sep='|', low_memory=False)

# Metrics
df['HasClaim'] = df['TotalClaims'] > 0
claim_frequency = df.groupby('HasClaim').size() / len(df)
claim_severity = df[df['HasClaim']]['TotalClaims'].mean()
df['Margin'] = df['TotalPremium'] - df['TotalClaims']

# Hypothesis 1: Risk differences across provinces
print("H0: No risk differences across provinces")
contingency_province = pd.crosstab(df['Province'], df['HasClaim'])
chi2_prov, p_prov, _, _ = chi2_contingency(contingency_province)
severity_prov = df[df['HasClaim']].groupby('Province')['TotalClaims'].mean()
print(f"Claim Frequency (Chi-squared): p-value = {p_prov:.4f}")
print(f"Claim Severity by Province:\n{severity_prov}")

# Hypothesis 2: Risk differences between zip codes
print("\nH0: No risk differences between zip codes")
top_zips = df['PostalCode'].value_counts().head(2).index
group_a = df[df['PostalCode'] == top_zips[0]]
group_b = df[df['PostalCode'] == top_zips[1]]
# Ensure groups are comparable
print("Checking group similarity (VehicleType):")
print(pd.crosstab(group_a['VehicleType'], group_b['VehicleType']))
freq_a = group_a['HasClaim'].mean()
freq_b = group_b['HasClaim'].mean()
t_stat_freq, p_freq = ttest_ind(group_a['HasClaim'], group_b['HasClaim'])
sev_a = group_a[group_a['HasClaim']]['TotalClaims']
sev_b = group_b[group_b['HasClaim']]['TotalClaims']
t_stat_sev, p_sev = ttest_ind(sev_a, sev_b, equal_var=False)
print(f"Claim Frequency (t-test): p-value = {p_freq:.4f}")
print(f"Claim Severity (t-test): p-value = {p_sev:.4f}")

# Hypothesis 3: Margin differences between zip codes
print("\nH0: No significant margin difference between zip codes")
t_stat_margin, p_margin = ttest_ind(group_a['Margin'], group_b['Margin'], equal_var=False)
print(f"Margin (t-test): p-value = {p_margin:.4f}")

# Hypothesis 4: Risk differences between Women and Men
print("\nH0: No significant risk difference between Women and Men")
df_gender = df[df['Gender'].isin(['Male', 'Female'])]
male = df_gender[df_gender['Gender'] == 'Male']
female = df_gender[df_gender['Gender'] == 'Female']
# Ensure groups are comparable
print("Checking group similarity (Province):")
print(pd.crosstab(male['Province'], female['Province']))
t_stat_gender_freq, p_gender_freq = ttest_ind(male['HasClaim'], female['HasClaim'])
sev_male = male[male['HasClaim']]['TotalClaims']
sev_female = female[female['HasClaim']]['TotalClaims']
t_stat_gender_sev, p_gender_sev = ttest_ind(sev_male, sev_female, equal_var=False)
print(f"Claim Frequency (t-test): p-value = {p_gender_freq:.4f}")
print(f"Claim Severity (t-test): p-value = {p_gender_sev:.4f}")

# Save results
with open('reports/ab_testing_results.txt', 'w') as f:
    f.write("A/B Testing Results\n")
    f.write(f"Province Risk: p-value = {p_prov:.4f}\n")
    f.write(f"Zip Code Frequency: p-value = {p_freq:.4f}\n")
    f.write(f"Zip Code Severity: p-value = {p_sev:.4f}\n")
    f.write(f"Zip Code Margin: p-value = {p_margin:.4f}\n")
    f.write(f"Gender Frequency: p-value = {p_gender_freq:.4f}\n")
    f.write(f"Gender Severity: p-value = {p_gender_sev:.4f}\n")
