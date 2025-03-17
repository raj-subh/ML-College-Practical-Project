import pandas as pd

# Load the dataset
data = pd.read_csv("customer_churn_dataset-training-master.csv")

print("Print Databset:\n", data.dtypes)

# Handle missing values (if any) - Drop rows with missing values to simplify
data = data.dropna()

# Label Encoding for the Gender column
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
data["Gender_LabelEncoded"] = label_encoder.fit_transform(data["Gender"])

# One-Hot Encoding for Gender and Subscription Type
data_onehot = pd.get_dummies(
    data, columns=["Gender", "Subscription Type"], drop_first=True
)

# Display the updated dataset
print("Dataset after Label Encoding for Gender:")
print(data[["Gender", "Gender_LabelEncoded"]].head())

print("\nDataset after One-Hot Encoding:")
print(data_onehot.head())
