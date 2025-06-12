
from docx import Document
from docx.shared import *
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document()

# Title Page
title = doc.add_heading('Interim Report: B5W3 Insurance Risk Analytics', 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph('Submitted for the Interim Deadline: June 13, 2025').alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph('Repository: https://github.com/moheranus/B5W3-Insurance-Risk-Analytics').alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph('Branch: task-2').alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph('Submitted on: June 12, 2025').alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph('By: Daniel Shobe').alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_page_break()

# Section 1: Overview
doc.add_heading('1. Overview', 1)
doc.add_paragraph(
    'This interim report summarizes Tasks 1 and 2 for the B5W3: End-to-End Insurance Risk Analytics & Predictive '
    'Modeling project. Task 1 established a Git repository, implemented CI/CD, and performed EDA. Task 2 set up a '
    'reproducible data pipeline using DVC.'
)

# Section 2: Task 1 Git Setup
doc.add_heading('2. Task 1: Git and GitHub Setup', 1)
doc.add_heading('2.1 Achievements', 2)
doc.add_paragraph(
    'Repository: https://github.com/moheranus/B5W3-Insurance-Risk-Analytics\n'
    'Branch: task-1 (merged into main)\n'
    'Structure:\n'
    '- data/: Hosts MachineLearningRating_v3.txt (~1M rows, 503.89 MB).\n'
    '- src/: Contains eda_insurance_analysis.py.\n'
    '- visualizations/: Stores 10 PNG visualizations, archived as visualizations.zip.\n'
    '- reports/: Includes this report.\n'
    '- .github/workflows/: Contains main.yml for CI/CD.\n'
    'Commits: Multiple on June 12, 2025.\n'
    'CI/CD: GitHub Actions workflow runs EDA script.'
)
doc.add_heading('2.2 KPIs', 2)
doc.add_paragraph(
    '- Dev Environment: Configured Python and Git.\n'
    '- Skills: Demonstrated Git, CI/CD, and data analysis.'
)

# Section 3: Task 1 EDA
doc.add_heading('3. Task 1: Exploratory Data Analysis & Statistics', 1)
doc.add_heading('3.1 Dataset', 2)
doc.add_paragraph(
    'File: MachineLearningRating_v3.txt (~1,000,098 rows, 52 columns).\n'
    'Period: February 2014–August 2015.\n'
    'Features: TotalPremium, TotalClaims, Province, VehicleType, Gender, make, TransactionMonth.'
)
doc.add_heading('3.2 EDA Execution', 2)
doc.add_paragraph(
    'Data Summarization:\n'
    '- TotalPremium: Mean 61.91, Std 230.28.\n'
    '- TotalClaims: Mean 64.86, Std 2,384.08.\n'
    '- Missing Values: NumberOfVehiclesInFleet (100%).\n'
    'Univariate Analysis: Skewed distributions.\n'
    'Bivariate Analysis: Loss ratios by Province.\n'
    'Temporal Trends: Seasonal patterns.\n'
    'Outlier Detection: TotalClaims outliers.\n'
    'Vehicle Make Analysis: Top 10 makes.'
)
doc.add_heading('3.3 Creative Visualizations', 2)
doc.add_paragraph(
    '- loss_ratio_province.png\n'
    '- monthly_trends.png\n'
    '- top_makes_claims.png'
)

# Section 4: Task 2 DVC
doc.add_heading('4. Task 2: DVC Setup', 1)
doc.add_heading('4.1 Achievements', 2)
doc.add_paragraph(
    'DVC Initialization: Ran dvc init.\n'
    'Local Remote Storage: Configured at C:\\dvc-storage.\n'
    'Data Tracking: Added MachineLearningRating_v3.txt with dvc add.\n'
    'Commits: Added .dvc files to Git.\n'
    'Data Push: Pushed to local remote.'
)
doc.add_heading('4.2 KPIs', 2)
doc.add_paragraph(
    '- Reproducibility: Ensured dataset versioning.\n'
    '- Skills: Demonstrated DVC proficiency.'
)

# Other Sections
doc.add_heading('5. Challenges and Solutions', 1)
doc.add_paragraph('Large File: Managed with DVC.\nPerformance: Sampled data.')
doc.add_heading('6. Next Steps', 1)
doc.add_paragraph('Task 3: A/B testing.\nTask 4: Modeling.')
doc.add_heading('7. Limitations', 1)
doc.add_paragraph('Negative values, sampled visualizations.')
doc.add_heading('8. Visualizations', 1)
doc.add_paragraph('In visualizations.zip on main.')
doc.add_heading('9. Conclusion', 1)
doc.add_paragraph('Tasks 1 and 2 delivered EDA and data pipeline.')

# Save
doc.save('reports/interim_report.docx')
