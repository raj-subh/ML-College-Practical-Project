# 3. Handling CSV Files:
# Import a CSV file into a DataFrame, explore its structure, and perform basic operations like renaming columns or finding missing values.


import pandas as pd
from sklearn.calibration import LabelEncoder

# Load the dataset
dataFrame = pd.read_csv("customer_churn_dataset-training-master.csv")

# Explore the structure of the DataFrame
print("Shape of DataFrame:", dataFrame.shape)
print("Columns in DataFrame:", dataFrame.columns)
print("Data Types of Columns:\n", dataFrame.dtypes)
print("First 5 Rows:\n", dataFrame.head())

# Renaming columns
# Explicitly reassign the updated DataFrame after renaming columns
dataFrame = dataFrame.rename(columns={"Usage Frequency": "UsageFrequency"})

# Finding missing values
# print("Missing Values in Each Column:\n", dataFrame.isnull().sum())

# # Fill numerical missing values with the mean
if "Age" in dataFrame.columns:
    dataFrame["Age"] = dataFrame["Age"].fillna(dataFrame["Age"].mean())

# check again there are any missing values
# print("Check Age Column Again:\n ", dataFrame.isnull().sum())

# # Fill categorical missing values with the mode
if "Gender" in dataFrame.columns:
    dataFrame["Gender"] = dataFrame["Gender"].fillna(dataFrame["Gender"].mode()[0])

# check gender column again
# print("Check Gender Column Again:\n ", dataFrame.isnull().sum())

# # Drop rows with missing values
dataFrame = dataFrame.dropna()

# # Drop columns with missing values
dataFrame = dataFrame.dropna(axis=1)

# # Final DataFrame shape
# print("Cleaned DataFrame Shape:", dataFrame.shape)

# Final dataset after pre-processing
print("Pre-Processed DataFrame:\n", dataFrame.head())
