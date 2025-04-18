import pandas as pd
from datetime import datetime

# Load the dataset
df = pd.read_csv("D:/ML_Practice/marketing_campaign.csv", sep="\t")

print("Available columns:\n", df.columns.tolist())


# Drop irrelevant columns
cols_to_drop = ["ID", "Z_CostContact", "Z_Revenue"]
existing_cols = [col for col in cols_to_drop if col in df.columns]
df.drop(columns=existing_cols, inplace=True)

# Convert 'Dt_Customer' to datetime format
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y")

# Fill missing values in 'Income' with the median
df["Income"] = df["Income"].fillna(df["Income"].median())

# Feature Engineering
df["Customer_Age"] = datetime.now().year - df["Year_Birth"]
df["Children"] = df["Kidhome"] + df["Teenhome"]
df["TotalSpend"] = df[
    [
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
    ]
].sum(axis=1)

# Drop original columns used to create new features
df.drop(columns=["Year_Birth", "Kidhome", "Teenhome"], inplace=True)

# Check final result
print(df.info())
print(df.head())

df.to_csv("D:/ML_Practice/PraticalFile/marketing_campaign_cleaned.csv", index=False)
