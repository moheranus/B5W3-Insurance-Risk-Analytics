import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import shap
import matplotlib.pyplot as plt

# Load prepared data
X_train_sev = pd.read_csv('data/X_train_severity.csv')
X_test_sev = pd.read_csv('data/X_test_severity.csv')
y_train_sev = pd.read_csv('data/y_train_severity.csv').values.ravel()
y_test_sev = pd.read_csv('data/y_test_severity.csv').values.ravel()

X_train_class = pd.read_csv('data/X_train_class.csv')
X_test_class = pd.read_csv('data/X_test_class.csv')
y_train_class = pd.read_csv('data/y_train_class.csv').values.ravel()
y_test_class = pd.read_csv('data/y_test_class.csv').values.ravel()

# Claim Severity Models
models_sev = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
}

results_sev = {}
for name, model in models_sev.items():
    model.fit(X_train_sev, y_train_sev)
    y_pred = model.predict(X_test_sev)
    rmse = np.sqrt(mean_squared_error(y_test_sev, y_pred))
    r2 = r2_score(y_test_sev, y_pred)
    results_sev[name] = {'RMSE': rmse, 'R-squared': r2}

# Claim Probability Models
models_class = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight=sum(y_train_class==0)/sum(y_train_class==1))
}

results_class = {}
for name, model in models_class.items():
    model.fit(X_train_class, y_train_class)
    y_pred = model.predict(X_test_class)
    acc = accuracy_score(y_test_class, y_pred)
    prec = precision_score(y_test_class, y_pred, zero_division=0)
    rec = recall_score(y_test_class, y_pred)
    f1 = f1_score(y_test_class, y_pred)
    results_class[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-score': f1}

# Feature Importance (SHAP) for XGBoost severity model
xgb_sev = models_sev['XGBoost']
explainer_sev = shap.TreeExplainer(xgb_sev)
shap_values_sev = explainer_sev.shap_values(X_test_sev)
shap.summary_plot(shap_values_sev, X_test_sev, show=False)
plt.savefig('reports/shap_severity.png')
plt.close()

# Feature Importance (SHAP) for XGBoost classification model
xgb_class = models_class['XGBoost']
explainer_class = shap.TreeExplainer(xgb_class)
shap_values_class = explainer_class.shap_values(X_test_class)
shap.summary_plot(shap_values_class, X_test_class, show=False)
plt.savefig('reports/shap_classification.png')
plt.close()

# Save results
with open('reports/model_results.txt', 'w') as f:
    f.write("Claim Severity Model Results:\n")
    for name, metrics in results_sev.items():
        f.write(f"{name}: RMSE={metrics['RMSE']:.2f}, R-squared={metrics['R-squared']:.4f}\n")
    f.write("\nClaim Probability Model Results:\n")
    for name, metrics in results_class.items():
        f.write(f"{name}: Accuracy={metrics['Accuracy']:.4f}, Precision={metrics['Precision']:.4f}, Recall={metrics['Recall']:.4f}, F1-score={metrics['F1-score']:.4f}\n")

print("Modeling complete. Results saved to reports/model_results.txt.")
print("SHAP plots saved to reports/shap_severity.png and reports/shap_classification.png.")