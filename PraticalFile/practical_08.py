# Linear Regression:
# Build a simple linear regression model to predict house prices or student grades.Visualize the
# regression line on a scatter plot.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 2: Generate Sample Data (Study Hours vs. Student Grades)
np.random.seed(42)

# Generate 50 random study hours between 1 and 10
study_hours = np.random.uniform(1, 10, 50)

# Generate corresponding grades with some random noise
grades = 5 * study_hours + np.random.normal(0, 5, 50)

# Convert to DataFrame
df = pd.DataFrame({"Study Hours": study_hours, "Student Grades": grades})

print(df.head())

# Step 3: Visualizing Data (Scatter Plot)
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df["Study Hours"], y=df["Student Grades"], color="blue")
plt.xlabel("Study Hours")
plt.ylabel("Student Grades")
plt.title("Study Hours vs. Student Grades")
plt.show()

# Step 4: Splitting Data for Training and Testing
X = df[["Study Hours"]]  # Independent variable
y = df[["Student Grades"]]  # Dependent variable

# Splitting dataset: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Training the Linear Regression Model
model = LinearRegression()  # Create model
model.fit(X_train, y_train)  # Train the model

# Step 6: Getting Model Parameters
m = model.coef_.item()  # Extract the single value properly
b = model.intercept_.item()  # Extract the single value properly

print(f"Equation: Grade = {m:.2f} * Study Hours + {b:.2f}")

# Step 7: Making Predictions
y_pred = model.predict(X_test)

# Step 8: Evaluating the Model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Step 9: Visualizing the Regression Line
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x=df["Study Hours"], y=df["Student Grades"], color="blue", label="Actual Data"
)
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Regression Line")
plt.xlabel("Study Hours")
plt.ylabel("Student Grades")
plt.title("Linear Regression: Study Hours vs. Student Grades")
plt.legend()
plt.show()

"""
Understanding the Output:
First Plot (Scatter Plot - Study Hours vs. Student Grades)

This plot represents the raw data points.
Each blue dot indicates a student's study hours (X-axis) and their corresponding grades (Y-axis).
You can observe a general upward trend, meaning students who study more tend to have higher grades.
Second Plot (Linear Regression Applied)

The blue dots still represent the actual data points.
A red regression line has been added.
The red line is the "best fit" line, showing the general trend in the data.
This line helps predict a student's grade based on their study hours.
The slope of the line indicates the rate at which grades increase with more study hours.
Key Takeaways:
The regression line is a mathematical representation of the relationship between study hours and grades.
If a student's study hours are known, their expected grade can be estimated using this line.
The closer the points are to the line, the better the linear model fits the data.
"""
