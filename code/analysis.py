# data cleaning/feature engineering
import pandas as pd
ds_jobs = pd.read_csv('code\Cleaned_DS_Jobs.csv', encoding='utf-8')
print(ds_jobs.head())

print(ds_jobs.columns)

#print(ds_jobs.isnull().sum())
#print(ds_jobs['seniority'].unique())

# Example keyword lists
soft_skills = ['communication', 'leadership', 'collaborate', 'team', 'adapt', 'problem solving', 'critical thinking']
technical_skills = ['SQL', 'Java', 'C\+\+', 'Scala', 'SAS', 'Git', 'Linux', 'Matlab', 'Azure', ' R,']

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
# Filter out 'manager' and 'director' rows
ds_jobs = ds_jobs[~ds_jobs['job_simp'].isin(['manager', 'director'])]

# Verify the updated dataset
print(ds_jobs['job_simp'].value_counts())


"""
# model

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
"""



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Encode the target variable
le = LabelEncoder()
ds_jobs['job_simp_encoded'] = le.fit_transform(ds_jobs['job_simp'])

# Define features (X) and target (y)
X = ds_jobs[skill_columns]
y = ds_jobs['job_simp_encoded']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(class_weight = {0: 5, 1: 5, 2: 1, 3: 5, 4: 4}, random_state=42)
rf.fit(X_train, y_train)

# Predict on test data
y_pred = rf.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Decode predictions back to original labels if needed
y_pred_labels = le.inverse_transform(y_pred)

print(ds_jobs['job_simp'].unique())
# Assuming your dataframe is named df and the column is 'job_simp'
job_counts = ds_jobs['job_simp'].value_counts()

print(job_counts)

importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Display top features
print("Random Forest Feature Importance:")
print(feature_importance_df)

import shap
import pandas as pd
import numpy as np

# Create SHAP explainer
explainer = shap.TreeExplainer(rf)

# Generate SHAP values
shap_values = explainer.shap_values(X_test)

# Debugging shapes
print("shap_values shape:", np.array(shap_values).shape)  # Should be (n_classes, n_samples, n_features)
print("X_test shape:", X_test.shape)                     # Should be (n_samples, n_features)

# Loop through classes
for i, class_name in enumerate(le.classes_):
    # Select SHAP values for the current class
    shap_class_values = shap_values[i]  # Should have shape (n_samples, n_features)

    # Debug shape of the current class's SHAP values
    print(f"Class {class_name} SHAP values shape:", shap_class_values.shape)
    
    # Ensure the shape matches (n_samples, n_features)
    if shap_class_values.shape != X_test.shape:
        print(f"Reshaping SHAP values for class {class_name}...")
        shap_class_values = shap_class_values.T  # Likely cause is the transpose is needed

    # Plot summary
    print(f"SHAP summary for class {class_name} (Index {i}):")
    shap.summary_plot(shap_class_values, X_test, plot_type="bar")




"""

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Step 3: Train a Random Forest classifier on the resampled data
rf_classifier = RandomForestClassifier(class_weight = {0: 3, 1: 3, 2: 1, 3: 5, 4: 5} ,random_state=42)
rf_classifier.fit(X_train_resampled, y_train_resampled)

# Step 4: Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Step 5: Evaluate the model's performance
print('SMOTE')
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

importances = rf_classifier.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Display top features
print("Random Forest (SMOTE) Feature Importance:")
print(feature_importance_df)
"""