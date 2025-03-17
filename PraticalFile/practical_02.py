# 2. Working with DataFrames:
# Create and manipulate a DataFrame using Pandas (e.g., adding/removing columns, filtering rows).

import pandas as pd

# Create a DataFrame
data = {
    "Name": ["Alice", "Bob", "Charlie", "David", "Eva"],
    "Age": [25, 35, 40, 45, 50],
    "City": ["New york", "Los Anglese", "Chicago", "House", "Phonix"],
    "Salary": [50000, 60000, 70000, 80000, 900000],
}

df = pd.DataFrame(data)
print("Original DataFrame: ")
print(df)

# Step 2: Add a new column
df["Department"] = ["HR", "Finance", "IT", "Marketing", "Sales"]
print("\nDataFrame after adding new column (Department): ")
print(df)

# Step 3: Remove a column
df = df.drop("City", axis=1)
print("\nDataFrame after removing the column (City): ")
print(df)

# Step 4: Filter rows where Salary > 60000
filtered_df = df[df["Salary"] > 60000]
print("\nFiltered DataFrame (Salary > 60000): ")
print(filtered_df)

# Step 5: Update a column (Increase Salary by 10%)
df["Salary"] = df["Salary"] * 1.10
print("\nDataFrame after increasing Salary by 10%: ")
print(df)

# Step 6: Add a new column like PhoneNo
df["PhoneNo"] = ["7044908354", "8084732296", "7488271195", "9679845656", "8985768459"]
print("\nDataFrame ater adding a new column (PhoneNo): ")
print(df)
