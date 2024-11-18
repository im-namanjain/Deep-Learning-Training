import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from feature_engine.outliers import ArbitraryOutlierCapper

import warnings
warnings.filterwarnings("ignore")

# Load the dataset
df = pd.read_csv("insurance.csv")

# Display basic information about the dataset

print(df.info())
print(df.describe())
print(df.isnull().sum())

# Visualizing categorical features

features = ['sex', 'smoker', 'region']

plt.figure(figsize=(20, 10))
for i, col in enumerate(features):
    plt.subplot(1, 3, i + 1)
    x = df[col].value_counts()
    plt.pie(x.values, labels=x.index, autopct='%1.1f%%')
    plt.title(f'Distribution of {col}')
plt.show()

# Grouped bar plots for selected features against charges

features = ['sex', 'children', 'smoker', 'region']

plt.figure(figsize=(20, 10))
for i, col in enumerate(features):
    plt.subplot(2, 2, i + 1)
    df.groupby(col)['charges'].mean().plot(kind='bar')
    plt.title(f'Mean charges by {col}')
    plt.ylabel('Mean charges')
plt.tight_layout()
plt.show()

# Scatter plots for age and bmi against charges, colored by smoker

features = ['age', 'bmi']

plt.figure(figsize=(17, 7))
for i, col in enumerate(features):
    plt.subplot(1, 2, i + 1)
    sns.scatterplot(data=df, x=col, y='charges', hue='smoker')
    plt.title(f'Scatter plot of {col} vs charges (colored by smoker)')
plt.tight_layout()
plt.show()

# Remove duplicates if any

df.drop_duplicates(inplace=True)

# Boxplots to visualize outliers for age and bmi

plt.figure(figsize=(10, 5))
sns.boxplot(x=df['age'])
plt.title('Boxplot of Age')
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(x=df['bmi'])
plt.title('Boxplot of BMI')
plt.show()

# Handling outliers using ArbitraryOutlierCapper from feature_engine

Q1 = df['bmi'].quantile(0.25)
Q3 = df['bmi'].quantile(0.75)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR
print("Lower Limit:", lower_limit)
print("Upper Limit:", upper_limit)

arb = ArbitraryOutlierCapper(min_capping_dict={'bmi': lower_limit}, max_capping_dict={'bmi': upper_limit})
df[['bmi']] = arb.fit_transform(df[['bmi']])

# Check skewness after capping outliers

print("Skewness of BMI after outlier capping:", df['bmi'].skew())

# Mapping categorical variables to numeric values

df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
df['region'] = df['region'].map({'northwest': 0, 'northeast': 1, 'southeast': 2, 'southwest': 3})

# Calculate correlation matrix

correlation_matrix = df.corr()
print("Correlation Matrix:")
print(correlation_matrix)