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

# evaluation