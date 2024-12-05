# data cleaning/feature engineering
import pandas as pd
ds_jobs = pd.read_csv('code\Cleaned_DS_Jobs.csv', encoding='utf-8')
print(ds_jobs.head())

print(ds_jobs.columns)

#print(ds_jobs.isnull().sum())
#print(ds_jobs['seniority'].unique())

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

print(skill_counts)

#print(ds_jobs.dtypes)
ds_jobs = ds_jobs[~ds_jobs['job_simp'].isin(['manager', 'director'])]

print(ds_jobs['job_simp'].value_counts())

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

le = LabelEncoder()
ds_jobs['job_simp_encoded'] = le.fit_transform(ds_jobs['job_simp'])

X = ds_jobs[skill_columns]
y = ds_jobs['job_simp_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(class_weight = {0: 5, 1: 5, 2: 1, 3: 5, 4: 4}, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

y_pred_labels = le.inverse_transform(y_pred)

print(ds_jobs['job_simp'].unique())

job_counts = ds_jobs['job_simp'].value_counts()

print(job_counts)

importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)


print("Random Forest Feature Importance:")
print(feature_importance_df)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel("Gini Importance")
plt.ylabel("Feature")
plt.title("Feature Importance (Gini)")
plt.gca().invert_yaxis()
plt.show()

import shap
import pandas as pd
import numpy as np


explainer = shap.TreeExplainer(rf)

shap_values = explainer.shap_values(X_test)

print("shap_values shape:", np.array(shap_values).shape)
print("X_test shape:", X_test.shape)

for i, class_name in enumerate(le.classes_):
    shap_class_values = shap_values[i]
    print(f"Class {class_name} SHAP values shape:", shap_class_values.shape)

    if shap_class_values.shape != X_test.shape:
        print(f"Reshaping SHAP values for class {class_name}...")
        shap_class_values = shap_class_values.T
    print(f"SHAP summary for class {class_name} (Index {i}):")
    shap.summary_plot(shap_class_values, X_test, plot_type="bar")
