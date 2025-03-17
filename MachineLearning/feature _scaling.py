import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Sample dataset
data = {"Age": [20, 25, 30, 35, 40, 50]}
df = pd.DataFrame(data)

# Initialize scalers
minmax_scaler = MinMaxScaler()
zscore_scaler = StandardScaler()

# Apply Min-Max Scaling
df["Age_MinMax"] = minmax_scaler.fit_transform(df[["Age"]])

# Apply Z-score Standardization
df["Age_Zscore"] = zscore_scaler.fit_transform(df[["Age"]])

print(df)
