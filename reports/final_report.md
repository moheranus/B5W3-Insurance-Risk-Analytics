# Dynamic Risk-Based Pricing for ACIS: A Data-Driven Approach

*By Daniel Shobe*  
*Submitted: June 17, 2025*  
*Repository: [https://github.com/moheranus/B5W3-Insurance-Risk-Analytics](https://github.com/moheranus/B5W3-Insurance-Risk-Analytics)*

## Executive Summary
For ACIS, we developed a risk-based pricing system using a 1M-row insurance dataset. Key deliverables:
- **EDA**: Identified sparse claims (2,788/1M rows) and regional risk patterns.
- **A/B Testing**: Confirmed province and zip code as risk drivers.
- **Predictive Models**: Built models for claim severity (Random Forest, RMSE: 8,280.56) and claim probability (XGBoost, F1-score: 0.0321) to optimize premiums.
- **Recommendations**: Adjust pricing by province/zip code, maintain gender-neutral policies, improve data collection.

This post details our approach, insights, and strategies to enhance ACIS’s profitability and fairness.

## Analytical Approach
1. **EDA (Task 1)**: Visualized premium/claim trends in `MachineLearningRating_v3.txt`.
2. **Data Pipeline (Task 2)**: Used DVC for data tracking.
3. **A/B Testing (Task 3)**: Tested risk differences via Chi-squared/t-tests.
4. **Predictive Modeling (Task 4)**:
   - **Claim Severity**: Linear Regression, Random Forest, XGBoost on `TotalClaims` (claims > 0).
   - **Claim Probability**: Random Forest, XGBoost for `HasClaim` (binary).
   - **Metrics**: RMSE, R-squared (regression); accuracy, precision, recall, F1-score (classification).
   - **Interpretability**: SHAP for feature importance.

## Key Insights
### EDA
- **Sparse Data**: 2,788 claims (0.28%), 108 Male/Female claims.
- **Severity Variance**: Ranges from 11,186 (Northern Cape) to 32,265 (Free State).

### A/B Testing
- **Province**: p=0.0000, reject H₀. Free State’s 45% higher loss ratio than Gauteng.
- **Zip Code**: p=0.0000 (frequency), reject H₀. Zip code 122 has 19% higher claim frequency than 2000.
- **Gender**: p=nan (frequency), inconclusive due to sparse data.

### Predictive Modeling
- **Claim Severity**:
  - **Results**:
    - Random Forest: RMSE=8,280.56, R-squared=0.9574 (best model).
    - XGBoost: RMSE=9,973.59, R-squared=0.9381.
    - Linear Regression: RMSE=39,531.58, R-squared=0.0283.
  - **SHAP Insights** (Random Forest):
    - `TotalPremium`: Higher premiums increase predicted claims by ~5,000 Rand per 10,000 Rand premium.
    - `IsHighRiskProvince`: Free State/KwaZulu-Natal add ~7,500 Rand to claims.
    - `PolicyAge`: Each year increases claims by ~1,200 Rand.
    - `Province_encoded`: High-risk provinces drive higher severity.
    - `VehicleType_encoded`: Heavy commercial vehicles increase claims.
  - **Impact**: Adjust premiums in high-risk provinces and for older policies.

- **Claim Probability**:
  - **Results**:
    - XGBoost: Accuracy=0.8646, Precision=0.0164, Recall=0.8047, F1-score=0.0321 (best model).
    - Random Forest: Accuracy=0.9361, Precision=0.0164, Recall=0.3710, F1-score=0.0314.
  - **SHAP Insights** (XGBoost):
    - `IsHighRiskProvince`: Increases claim probability by ~0.15%.
    - `PostalCode`: High-frequency zip codes (e.g., 122) raise likelihood by ~0.1%.
    - `TotalPremium`: Higher premiums correlate with ~0.05% higher probability.
    - `PolicyAge`: Older policies increase probability.
    - `VehicleType_encoded`: Commercial vehicles raise risk.
  - **Impact**: Use `P(Claim) * Severity + 10% Loading + 5% Margin` for pricing.

## Pricing and Marketing Strategies
1. **Province Pricing**:
   - Increase premiums in Free State by 20–30% (32,265 severity, ~7,500 Rand higher predicted claims).
   - Reduce in Northern Cape (11,186 severity) to attract customers.

2. **Zip Code Pricing**:
   - Raise premiums in zip code 122 by 10–15% (0.43% frequency, ~0.1% higher probability).
   - Analyze additional zip codes as data grows.

3. **Gender-Neutral Pricing**:
   - Maintain due to inconclusive results (108 claims).
   - Improve gender data collection (940,990 “Not specified”).

4. **Risk-Based Premium**:
   - Formula: `Premium = P(Claim) * Severity + 10% Loading + 5% Margin`.
   - Example: Free State policy (0.5% probability, 30,000 severity) = (0.005 * 30,000) + 15 + 7.5 = 172.5 Rand.

5. **Marketing**:
   - Target low-risk provinces (e.g., Northern Cape) with competitive pricing.
   - Promote transparent, data-driven pricing to build trust.

## Limitations and Future Work
- **Sparse Data**: 2,788 claims (0.28%) limit gender analysis and classification precision.
- **Features**: Add vehicle age, driver experience for better predictions.
- **Tuning**: Optimize Random Forest/XGBoost hyperparameters to reduce RMSE.
- **Future**:
   - Collect more claims data, especially for gender.
   - Implement real-time pricing.
   - Explore deep learning for complex patterns.

## Conclusion
Our Random Forest (severity) and XGBoost (probability) models enable ACIS to implement precise, fair pricing. Province/zip code adjustments, gender-neutral policies, and data improvements will drive profitability and customer trust.

*Acknowledgments*: Thanks to ACIS for the dataset and guidance.

*Submitted: June 17, 2025*