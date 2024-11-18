# data cleaning/feature engineering
import pandas as pd
ds_jobs = pd.read_csv('code\Cleaned_DS_Jobs.csv', encoding='utf-8')
print(ds_jobs.head())

print(ds_jobs.columns)

#print(ds_jobs.isnull().sum())
#print(ds_jobs['seniority'].unique())

# Example keyword lists
soft_skills = ['communication', 'leadership', 'collaboration', 'teamwork', 'adapt', 'problem-solving', 'critical thinking']
technical_skills = ['SQL', 'Java', 'C\+\+', 'Scala', 'SAS', 'Git', 'Linux', 'Matlab', 'Azure']

# Add soft skill columns
for skill in soft_skills:
    ds_jobs[f'{skill}'] = ds_jobs['Job Description'].str.contains(skill, case=False, na=False).astype(int)

# Add technical skill columns
for skill in technical_skills:
    ds_jobs[f'{skill}'] = ds_jobs['Job Description'].str.contains(skill, case=False, na=False).astype(int)

print(ds_jobs.head())

# Filter the skill-related columns
skill_columns = soft_skills + technical_skills  + ['python',
       'excel', 'hadoop', 'spark', 'aws', 'tableau', 'big_data']# Already lists of column names

# Count the number of rows where each skill is mentioned
skill_counts = ds_jobs[skill_columns].sum()

# Display the counts
print(skill_counts)

#print(ds_jobs.dtypes)
# eda


# model
'''
# Create binary target variable
ds_jobs['is_data_scientist'] = (ds_jobs['job_simp'] == 'data scientist').astype(int)

from sklearn.linear_model import LogisticRegression

# Features and target
X = ds_jobs[skill_columns]
y = ds_jobs['is_data_scientist']

# Fit logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Get feature importance
importance = pd.Series(model.coef_[0], index=skill_columns).sort_values(ascending=False)
print("Feature importance for predicting 'Data Scientist' jobs:")
print(importance)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Features and target
X = ds_jobs[skill_columns]
y = ds_jobs['job_simp']  # Assume this column categorizes job titles

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Feature importance
importance = pd.Series(rf.feature_importances_, index=skill_columns).sort_values(ascending=False)
print("Skill importance across different data science jobs:")
print(importance)

# Create a binary target for "Data Scientist" jobs
ds_jobs['is_data_scientist'] = (ds_jobs['job_simp'] == 'data scientist').astype(int)

# Fit a Random Forest for this binary classification
rf_ds = RandomForestClassifier(random_state=42)
rf_ds.fit(X, ds_jobs['is_data_scientist'])

# Feature importance for "Data Scientist" roles
importance_ds = pd.Series(rf_ds.feature_importances_, index=skill_columns).sort_values(ascending=False)
print("Skill importance for Data Scientist roles:")
print(importance_ds)
'''
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# One-hot encoding using pandas
job_simp_encoded = pd.get_dummies(ds_jobs['job_simp'], prefix='job_simp', drop_first=True)

# Adding the encoded columns to the original dataset
ds_jobs_encoded = pd.concat([ds_jobs, job_simp_encoded], axis=1)

print(ds_jobs_encoded.head())

# For binary classification, create a single target column
y = (ds_jobs['job_simp'] == 'data scientist').astype(int)
X = ds_jobs[skill_columns]


 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Logistic Regression
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:, 1]  # Predicted probabilities for ROC-AUC

# Extract coefficients
coefficients = lr.coef_[0]  # For binary classification, take the first set of coefficients
feature_importance_df_lr = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': coefficients
}).sort_values(by='Coefficient', ascending=False)

# Display coefficients
print("Logistic Regression Coefficients:")
print(feature_importance_df_lr)

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]  # Predicted probabilities for ROC-AUC

importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Display top features
print("Random Forest Feature Importance:")
print(feature_importance_df)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

# Logistic Regression
print("Logistic Regression Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_lr):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_lr):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_lr):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_lr):.4f}")

# Random Forest
print("\nRandom Forest Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_rf):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_rf):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_rf):.4f}")
# Logistic Regression
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))

# Random Forest
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# Logistic Regression ROC Curve
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
# Random Forest ROC Curve
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label="Logistic Regression (AUC = {:.4f})".format(roc_auc_score(y_test, y_prob_lr)))
plt.plot(fpr_rf, tpr_rf, label="Random Forest (AUC = {:.4f})".format(roc_auc_score(y_test, y_prob_rf)))
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
