# 4. Data Pre-processing:
# Handle missing data in a dataset (e.g., filling with mean/median or dropping rows).
# Encode categorical data using one-hot encoding or label encoding.

import pandas as pd

# Load the dataset
dataFrame = pd.read_csv("customer_churn_dataset-training-master.csv")

# Display initial missing values
print("Missing Values Before Handling:\n", dataFrame.isnull().sum())

# Handling missing values
# Fill numerical columns with mean
for col in dataFrame.select_dtypes(include=["number"]).columns:
    dataFrame[col] = dataFrame[col].fillna(dataFrame[col].mean())

# Fill categorical columns with mode
for col in dataFrame.select_dtypes(include=["object"]).columns:
    dataFrame[col] = dataFrame[col].fillna(dataFrame[col].mode()[0])

# Display missing values after handling
print("\nMissing Values After Handling:\n", dataFrame.isnull().sum())

# Save cleaned data to a new file
cleaned_file_path = "d:/ML_Practice/PraticalFile/customer_churn_dataset_cleaned.csv"
dataFrame.to_csv(cleaned_file_path, index=False)

print(
    "\n✅ Missing data handled successfully. Cleaned file saved at:", cleaned_file_path
)
