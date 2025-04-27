import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# Set the style for visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set2')

# Create output directories if they don't exist
os.makedirs('plots', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Load the German Credit Data
print("Loading the German Credit Data...")
data = pd.read_csv('german_credit_data.csv', index_col=0)

# Display basic information about the dataset
print("\nDataset Information:")
print(f"Shape: {data.shape}")
print("\nFirst few rows:")
print(data.head())

print("\nData types:")
print(data.dtypes)

print("\nSummary statistics:")
print(data.describe())

# Check for missing values
print("\nMissing values:")
missing_values = data.isnull().sum()
print(missing_values[missing_values > 0])
missing_percentage = (missing_values / len(data)) * 100
print("\nMissing values percentage:")
print(missing_percentage[missing_percentage > 0])

# Visualize missing values
plt.figure(figsize=(10, 6))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.tight_layout()
plt.savefig('plots/missing_values_heatmap.png')

# Exploratory Data Analysis
print("\nPerforming Exploratory Data Analysis...")

# Distribution of categorical variables
categorical_cols = ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']

fig, axes = plt.subplots(3, 2, figsize=(15, 15))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    sns.countplot(x=col, data=data, ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('plots/categorical_distributions.png')

# Distribution of numerical variables
numerical_cols = ['Age', 'Credit amount', 'Duration']

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, col in enumerate(numerical_cols):
    sns.histplot(data[col], kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)

plt.tight_layout()
plt.savefig('plots/numerical_distributions.png')

# Boxplots for numerical variables to detect outliers
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, col in enumerate(numerical_cols):
    sns.boxplot(y=data[col], ax=axes[i])
    axes[i].set_title(f'Boxplot of {col}')

plt.tight_layout()
plt.savefig('plots/numerical_boxplots.png')

# Correlation matrix
plt.figure(figsize=(10, 8))
corr_matrix = data.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('plots/correlation_matrix.png')

# Relationship between categorical variables and credit amount
fig, axes = plt.subplots(3, 2, figsize=(15, 15))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    sns.boxplot(x=col, y='Credit amount', data=data, ax=axes[i])
    axes[i].set_title(f'{col} vs Credit amount')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('plots/categorical_vs_credit_amount.png')

# Data Preprocessing
print("\nPreprocessing the data...")

# Handle missing values
# For categorical variables: fill with the most frequent value
# For numerical variables: fill with the median

# First, let's create a copy of the data
processed_data = data.copy()

# Impute missing values
for col in categorical_cols:
    if missing_values[col] > 0:
        most_frequent = processed_data[col].mode()[0]
        processed_data[col].fillna(most_frequent, inplace=True)
        print(f"Filled missing values in {col} with '{most_frequent}'")

for col in numerical_cols:
    if missing_values[col] > 0:
        median_value = processed_data[col].median()
        processed_data[col].fillna(median_value, inplace=True)
        print(f"Filled missing values in {col} with median: {median_value}")

# Check if there are any remaining missing values
print("\nRemaining missing values:")
print(processed_data.isnull().sum().sum())

# Feature Engineering
print("\nPerforming Feature Engineering...")

# Create age groups
processed_data['Age_Group'] = pd.cut(processed_data['Age'], 
                                   bins=[0, 25, 35, 45, 55, 65, 100],
                                   labels=['<25', '25-35', '35-45', '45-55', '55-65', '>65'])

# Create credit amount groups
processed_data['Credit_Amount_Group'] = pd.cut(processed_data['Credit amount'],
                                             bins=[0, 1000, 2000, 5000, 10000, 20000],
                                             labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

# Create duration groups
processed_data['Duration_Group'] = pd.cut(processed_data['Duration'],
                                        bins=[0, 12, 24, 36, 48, 72],
                                        labels=['<1yr', '1-2yrs', '2-3yrs', '3-4yrs', '>4yrs'])

# One-hot encode categorical variables
print("\nOne-hot encoding categorical variables...")

# Define categorical columns to encode
categorical_cols_to_encode = categorical_cols + ['Age_Group', 'Credit_Amount_Group', 'Duration_Group']

# Create a one-hot encoder
encoder = OneHotEncoder(sparse=False, drop='first')

# Fit and transform the categorical columns
encoded_data = encoder.fit_transform(processed_data[categorical_cols_to_encode])

# Get the feature names after one-hot encoding
feature_names = encoder.get_feature_names_out(categorical_cols_to_encode)

# Create a DataFrame with the encoded features
encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=processed_data.index)

# Combine the encoded features with the numerical features
numerical_data = processed_data[numerical_cols].copy()

# Standardize numerical features
scaler = StandardScaler()
numerical_data_scaled = scaler.fit_transform(numerical_data)
numerical_data_scaled_df = pd.DataFrame(numerical_data_scaled, 
                                       columns=numerical_cols,
                                       index=processed_data.index)

# Combine numerical and categorical data
final_data = pd.concat([numerical_data_scaled_df, encoded_df], axis=1)

print("\nFinal preprocessed data shape:", final_data.shape)
print("\nPreprocessed data columns:")
print(final_data.columns.tolist())

# Save the preprocessed data
print("\nSaving preprocessed data...")
final_data.to_csv('data/preprocessed_data.csv')
processed_data.to_csv('data/processed_data_with_features.csv')

print("\nPreprocessing complete! Files saved in the 'data' directory.")

# Display some visualizations of the preprocessed data
plt.figure(figsize=(12, 8))
sns.pairplot(processed_data[['Age', 'Credit amount', 'Duration', 'Sex']], hue='Sex')
plt.savefig('plots/pairplot.png')

# Visualize the distribution of credit amount by purpose
plt.figure(figsize=(12, 6))
sns.boxplot(x='Purpose', y='Credit amount', data=processed_data)
plt.title('Credit Amount by Purpose')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/credit_amount_by_purpose.png')

print("\nVisualization plots saved in the 'plots' directory.")