# Interim Report: B5W3 Insurance Risk Analytics

## Overview
This interim report summarizes Task 1 (Git Setup & Exploratory Data Analysis) for the B5W3: End-to-End Insurance Risk Analytics & Predictive Modeling project, submitted for the interim deadline of June 13, 2025. Task 1 established a Git repository, performed comprehensive EDA on the insurance dataset, and generated visualizations to uncover risk and profitability patterns, addressing key performance indicators (KPIs) and guiding questions.

## Task 1: Git and GitHub Setup

### Achievements
- **Repository**: [https://github.com/moheranus/B5W3-Insurance-Risk-Analytics](https://github.com/moheranus/B5W3-Insurance-Risk-Analytics)
- **Branch**: `task-1`
- **Structure**:
  - `data/`: Hosts `MachineLearningRating_v3.txt` (~1M rows, 503.89 MB).
  - `src/`: Contains `eda_insurance_analysis.py`.
  - `visualizations/`: Stores 10 PNG visualizations, archived as `visualizations.zip`.
  - `reports/`: Includes this report.
- **README**: Created with project overview, setup instructions, and usage.
- **Commits**: Multiple commits on June 12, 2025, including:
  - Visualizations archive (`visualizations.zip`).
  - `.gitignore` update to exclude large dataset.
  - EDA script updates.
- **CI/CD**: Planned for future tasks using GitHub Actions.

### KPIs
- **Dev Environment Setup**: Configured Python environment with pandas, seaborn, matplotlib; Git repository initialized.
- **Relevant Skills**: Demonstrated Git proficiency (branching, committing, history management) and data analysis skills.

## Task 1: Exploratory Data Analysis & Statistics

### Dataset
- **File**: `MachineLearningRating_v3.txt` (~1,000,098 rows, 52 columns).
- **Period**: February 2014–August 2015 (18 months).
- **Key Features**: `TotalPremium`, `TotalClaims`, `CustomValueEstimate`, `Province`, `VehicleType`, `Gender`, `make`, `TransactionMonth`.

### EDA Execution
The EDA was conducted using `src/eda_insurance_analysis.py`, addressing the guiding questions.

#### Data Summarization
- **Descriptive Statistics**:
  - `TotalPremium`: Mean 61.91, Std 230.28, Range [-782.58, 65,282.60].
  - `TotalClaims`: Mean 64.86, Std 2,384.08, Range [-12,002.41, 393,092.10].
  - `CustomValueEstimate`: Mean 225,531, Std 564,516, 220,456 non-null (78% missing).
  - `SumInsured`: Mean 604,173, Std 1,508,332.
  - `CalculatedPremiumPerTerm`: Mean 117.88, Std 399.70.
  - **Variability**: High standard deviations indicate skewed distributions, especially for `TotalClaims`.
- **Data Structure**:
  - 52 columns: 15 int64, 14 float64, 22 object, 1 bool.
  - `TransactionMonth` converted to datetime for temporal analysis.
  - Categorical variables (`Province`, `VehicleType`, `Gender`) properly formatted as strings.

#### Data Quality Assessment
- **Missing Values**:
  - `NumberOfVehiclesInFleet`: 100% missing (1,000,098 rows).
  - `CrossBorder`: 99.93% missing.
  - `CustomValueEstimate`: 77.96% missing.
  - `WrittenOff`, `Rebuilt`, `Converted`: 64.18% missing.
  - `NewVehicle`: 15.33%, `Bank`: 14.59%, `Gender`: 0.95%.
  - **Action**: Columns with >99% missing may be dropped; others require imputation.

#### Univariate Analysis
- **Numerical Distributions**: Histograms (`numerical_distributions.png`) show right-skewed distributions for `TotalPremium`, `TotalClaims`, and `CustomValueEstimate`.
- **Categorical Distributions**: Bar charts (`categorical_distributions.png`) reveal:
  - `Province`: Western Cape and Gauteng dominate.
  - `VehicleType`: Passenger vehicles most common.
  - `Gender`: Balanced distribution, with some missing values.

#### Bivariate/Multivariate Analysis
- **Loss Ratio (TotalClaims / TotalPremium)**:
  - Overall portfolio loss ratio calculated in `eda_insurance_analysis.py`.
  - Visualized by:
    - `Province` (`loss_ratio_province.png`): Varies significantly, with high ratios in certain regions.
    - `VehicleType` (`loss_ratio_vehicle_type.png`): Heavy vehicles show higher ratios.
    - `Gender` (`loss_ratio_gender.png`): Slight differences observed.
  - **Insight**: Identifies high-risk segments for pricing adjustments.
- **Correlations**: Heatmap (`correlation_matrix.png`) shows weak correlations between `TotalPremium` and `TotalClaims`.
- **Premium vs. Claims by ZipCode**: Scatter plot (`premium_vs_claims_postalcode.png`) highlights geographic risk clusters.

#### Temporal Trends
- **Analysis**: Monthly aggregation of `TotalClaims` and `TotalPremium` (`monthly_trends.png`).
- **Findings**:
  - Claim frequency and severity fluctuate over 18 months (Feb 2014–Aug 2015).
  - Seasonal patterns observed, with peaks in certain months.
  - **Insight**: Informs temporal risk modeling.

#### Outlier Detection
- **Box Plots** (`box_plots.png`):
  - `TotalClaims`: Significant outliers (e.g., claims up to 393,092.10).
  - `TotalPremium`: Outliers in high-premium policies.
  - `CustomValueEstimate`: Extreme values due to 78% missing data.
  - **Action**: Outliers may require capping or removal for modeling.

#### Vehicle Make/Model Analysis
- **Top Makes**: Bar chart (`top_makes_claims.png`) shows top 10 vehicle makes by average claim amount.
- **Findings**: Certain makes (e.g., luxury brands) have higher claims, indicating higher risk.

### Creative Visualizations
Three notable visualizations capturing key insights:
1. **Loss Ratio by Province** (`loss_ratio_province.png`): Bar plot with vibrant colors, highlighting regional risk differences.
2. **Monthly Trends** (`monthly_trends.png`): Dual-line plot showing claims and premiums over time, revealing seasonal patterns.
3. **Top Makes by Claims** (`top_makes_claims.png`): Bar plot emphasizing high-risk vehicle makes.

### Statistical Thinking
- **Distributions**: Used histograms with KDE to model numerical distributions, confirming skewness.
- **Correlations**: Pearson correlation matrix to assess relationships.
- **Loss Ratios**: Calculated as a key metric for profitability, analyzed across multiple dimensions.
- **References**: Self-learned EDA techniques from “Python for Data Analysis” (Wes McKinney) and seaborn documentation.

### KPIs
- **Proactivity**: Independently resolved large file issues (`git filter-repo`) and optimized EDA with sampling.
- **EDA Techniques**: Applied histograms, bar charts, scatter plots, heatmaps, and box plots.
- **Statistical Understanding**: Used appropriate distributions and visualizations to derive actionable insights (e.g., high-risk provinces, vehicle types).

## Challenges and Solutions
- **Large File**: Removed `MachineLearningRating_v3.txt` from Git history using `git filter-repo`.
- **Performance**: Sampled 10,000 rows for visualizations to handle ~1M rows.
- **Data Quality**: Identified negative values and missing data for future preprocessing.

## Task 2: DVC Setup
- **Status**: Planned.
- **Plan**: Initialize DVC, track dataset, set up local storage.

## Next Steps
- Task 5: DVC setup for data versioning.
- Task 3: A/B testing for risk assessment.
- Task 4: Predictive modeling.

## Limitations
- Negative `TotalPremium`/`TotalClaims` require cleaning.
- Visualizations use sampled data (10,000 rows).
- High missingness in `CustomValueEstimate` (78%) and other columns.
- Potential mixed types in `CapitalOutstanding` and `CrossBorder`.

## Visualizations
- Archived in `visualizations.zip` on `task-1` branch.

## Conclusion
Task 1 established a robust EDA pipeline, uncovering critical risk and profitability insights. The repository is ready for Task 2, with deliverables submitted for the interim deadline.

*Submitted: June 12, 2025*