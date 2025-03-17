# 1. Understanding the Basics of ML:
# Write a Python program to implement a simple "Hello ML World" example using libraries like NumPy or Pandas.


# Import necessary libraries
import numpy as np
import pandas as pd

# Step 1: Create a "Hello ML World" dataset using NumPy
# Generate a NumPy array with random data
np.random.seed(42)  # For reproducibility
data = np.random.rand(10, 3)  # 10 rows, 3 columns of random numbers between 0 and 1

# Step 2: Convert the NumPy array into a Pandas DataFrame
columns = ["Feature1", "Feature2", "Feature3"]
df = pd.DataFrame(data, columns=columns)

# Step 3: Add a new column with a simple operation (e.g., sum of other columns)
df["Target"] = df["Feature1"] + df["Feature2"] + df["Feature3"]

# Step 4: Print the DataFrame and display basic statistics
print("Hello ML World: Here is the dataset!")
print(df)

print("\nBasic Statistics:")
print(df.describe())

# Step 5: Display the top 5 rows to simulate exploratory data analysis
print("\nTop 5 rows of the dataset:")
print(df.head())
