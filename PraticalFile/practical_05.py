# 5. Feature Scaling:
# Normalize or standardize a dataset using Min-Max scaling or Z-score scaling.


import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# load dataset
dataFrame = pd.read_csv("customer_churn_dataset-training-master.csv")

# Select numeraical columns (Example: Age column)
if "Age" in dataFrame.columns:
    # Initialize scalers
    minmax_scaler = MinMaxScaler()
    zscore_scaler = StandardScaler()

    # Reshape the column as required by sklearn
    dataFrame["Age_MinMax"] = minmax_scaler.fit_transform(dataFrame[["Age"]])
    dataFrame["Age_Zscore"] = zscore_scaler.fit_transform(dataFrame[["Age"]])

    # Display first 5 rows
    print(dataFrame[["Age", "Age_MinMax", "Age_Zscore"]].head())

else:
    print("COlumn 'AGe' not found in dataset.")
