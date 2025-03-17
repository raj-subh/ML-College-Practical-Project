# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate random data using NumPy
np.random.seed(42)
data = np.random.rand(50, 2)  # 50 rows, 2 features

# Create a DataFrame using Pandas
df = pd.DataFrame(data, columns=["Feature1", "Feature2"])

# Add a target column
df["Target"] = df["Feature1"] + df["Feature2"]

# Plot the data
plt.scatter(df["Feature1"], df["Target"], label="Feature1 vs Target", color="blue")
plt.scatter(df["Feature2"], df["Target"], label="Feature2 vs Target", color="red")
plt.title("Hello ML World!")
plt.xlabel("Features")
plt.ylabel("Target")
plt.legend()
plt.show()
