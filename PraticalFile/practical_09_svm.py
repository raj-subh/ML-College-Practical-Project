# Build and Train LogisticRegression Model, Tuning model using SVM in ML

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("D:/ML_Practice/diabetes-dataset.csv")

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

print("\n Data Split Completed!")

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize & Train SVM with RBF Kernel (default)
model = SVC(kernel="rbf", C=1.0)

# Fit model to training data
model.fit(X_train_scaled, y_train)

print(" SVM Model Training Completed!")

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n SVM Model Accuracy: {accuracy * 100:.2f}%")

# Display classification report
print("\n Classification Report:")
print(classification_report(y_test, y_pred))

# Display confusion matrix
print("\n Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
