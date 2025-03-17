# Data Visualization:
# Create basic plots (line, bar, scatter) using Matplotlib and Seaborn.Visualize correlations in a
# dataset using heatmaps.

# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: creating a sample dataset
np.random.seed(42)
data = {
    "Month": [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ],
    "Product_A_Sales": np.random.randint(200, 500, 12),
    "Product_B_Sales": np.random.randint(100, 400, 12),
    "Product_C_Sales": np.random.randint(50, 350, 12),
}

df = pd.DataFrame(data)
print(df.head())

# Step 3: Line Plot (Sales Over Months)
# A line plot is useful for showing trends over time.
plt.figure(figsize=(10, 5))
plt.plot(
    df["Month"], df["Product_A_Sales"], marker="o", linestyle="-", label="Product A"
)
plt.plot(
    df["Month"], df["Product_B_Sales"], marker="s", linestyle="--", label="Product B"
)
plt.plot(
    df["Month"], df["Product_C_Sales"], marker="^", linestyle="-.", label="Product C"
)

plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Monthly Sales Trend")
plt.legend()
plt.grid(True)
plt.show()

# Step 4: Bar Plot (Comparing Monthly Sales)
# A bar plot is great for comparing values across categories.
df.plot(x="Month", kind="bar", figsize=(10, 5))
plt.title("Monthly Sales Comparison")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend(["Product A", "Product B", "Product C"])
plt.show()

# Step 5: Scatter Plot (Sales Relationship)
# A scatter plot is useful to see relationships between two variables.
plt.figure(figsize=(8, 5))
plt.scatter(
    df["Product_A_Sales"],
    df["Product_B_Sales"],
    color="blue",
    alpha=0.6,
    label="A vs B",
)
plt.scatter(
    df["Product_A_Sales"], df["Product_C_Sales"], color="red", alpha=0.6, label="A vs C"
)

plt.xlabel("Product A Sales")
plt.ylabel("Other Product Sales")
plt.title("Sales Relationship")
plt.legend()
plt.show()

# Step 6: Heatmap (Correlation Between Sales)
# A heatmap helps visualize correlation between different variables.
plt.figure(figsize=(8, 5))
sns.heatmap(
    df.drop("Month", axis=1).corr(), annot=True, cmap="coolwarm", linewidths=0.5
)
plt.title("Sales Correlation HeatMap")
plt.show()
