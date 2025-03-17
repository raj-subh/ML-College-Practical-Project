# Build and Train LogisticRegression Model

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("D:/ML_Practice/diabetes-dataset.csv")

# Display basic dataset info
print("\n Dataset Preview:")
print(df.head(), "\n")

print("\n Dataset Shape:")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n Checking for missing values:")
print(df.isnull().sum())

# Assuming 'Outcome' is the target variable
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

print("\n Features (X):")
print(X.head())

print("\n Target Variable (y):")
print(y.head())

# Splitting dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n✅ Data Split Completed!")

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the model
model = LogisticRegression(max_iter=500, solver="liblinear")
model.fit(X_train_scaled, y_train)

print("\n✅ Model Training Completed!")

# 6. Model Evaluation

# Predictions
y_pred = model.predict(X_test_scaled)

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"\n Model Accuracy: {accuracy * 100:.2f}%")

# Classification Report
print("\n Classification Report: ")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\n Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred))
