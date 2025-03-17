import pandas as pd

# Load the dataset
df = pd.read_csv("D:/ML_Practice/raw_sales.csv")

# Convert 'datesold' to datetime
df["datesold"] = pd.to_datetime(df["datesold"], errors="coerce")

# “errors='coerce' ensures that any invalid date values are converted to NaT (Not a Time) instead of causing an error.”
# Aggregate sales per day
daily_sales = df.groupby("datesold").agg({"price": "sum"}).reset_index()
date_range = pd.date_range(
    start=daily_sales["datesold"].min(), end=daily_sales["datesold"].max(), freq="D"
)

# Create a complete time series with all dates
full_data = pd.DataFrame({"datesold": date_range})

# Merge with existing sales data and fill missing dates with zero sales
full_data = full_data.merge(daily_sales, on="datesold", how="left").fillna({"price": 0})

# Save the cleaned dataset
full_data.to_csv("D:/ML_Practice/cleaned_sales.csv", index=False)
