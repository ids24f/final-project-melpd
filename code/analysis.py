# data cleaning/feature engineering
import pandas as pd
ds_jobs = pd.read_csv('code\Cleaned_DS_Jobs.csv', encoding='utf-8')
print(ds_jobs.head())

print(ds_jobs.columns)

#print(ds_jobs.isnull().sum())
#print(ds_jobs['seniority'].unique())

# Example keyword lists
soft_skills = ['communication', 'leadership', 'collaboration', 'teamwork', 'adaptability', 'problem-solving', 'critical thinking']
technical_skills = ['SQL', 'Java', 'C\+\+', 'Scala', 'AWS', 'Docker', 'Git', 'Linux', 'Matlab', 'Azure']

# Add soft skill columns
for skill in soft_skills:
    ds_jobs[f'{skill}'] = ds_jobs['Job Description'].str.contains(skill, case=False, na=False).astype(int)

# Add technical skill columns
for skill in technical_skills:
    ds_jobs[f'{skill}'] = ds_jobs['Job Description'].str.contains(skill, case=False, na=False).astype(int)

print(ds_jobs.head())

# Filter the skill-related columns
skill_columns = soft_skills + technical_skills  # Already lists of column names

# Count the number of rows where each skill is mentioned
skill_counts = ds_jobs[skill_columns].sum()

# Display the counts
print(skill_counts)

cpp_count = ds_jobs[ds_jobs['Job Description'].str.contains('C\+\+', case=False, na=False)].shape[0]
print(f'C++ mentions in unique job descriptions: {cpp_count}')

print(ds_jobs.dtypes)
# eda


# model


# evaluation