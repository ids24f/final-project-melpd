# data cleaning/feature engineering
import pandas as pd
ds_jobs = pd.read_csv('code\Cleaned_DS_Jobs.csv', encoding='utf-8')
print(ds_jobs.head())

print(ds_jobs.columns)

#print(ds_jobs.isnull().sum())
#print(ds_jobs['seniority'].unique())

# Example keyword lists
soft_skills = ['communication', 'leadership', 'collaboration', 'teamwork', 'adapt', 'problem-solving', 'critical thinking']
technical_skills = ['SQL', 'Java', 'C\+\+', 'Scala', 'Docker', 'Git', 'Linux', 'Matlab', 'Azure']

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
ds_jobs['is_data_scientist'] = ds_jobs['job_simp'].apply(lambda x: 1 if 'scientist' in x.lower() else 0)

from sklearn.linear_model import LogisticRegression

# Features and target
X = ds_jobs[soft_skills + technical_skills]
y = ds_jobs['is_data_scientist']

# Fit logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Get feature importance
importance = pd.Series(model.coef_[0], index=soft_skills + technical_skills).sort_values(ascending=False)
print("Feature importance for predicting 'Data Scientist' jobs:")
print(importance)



# evaluation