import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Creating a dataset
data = {
    "Height": [150, 160, 165, 170, 175, 180, 185, 190, 195, 200],
    "Weight": [50, 60, 65, 70, 75, 80, 85, 90, 95, 100],
    "Class": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],  # 0 = Non-Athlete, 1 = Athlete
}

df = pd.DataFrame(data)

# Splitting data into features (X) and target variable (y)
X = df[["Height", "Weight"]]
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create KNN classifier with K=3
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train, y_train)

# Predict on test data
y_pred = knn.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Plot the dataset
plt.scatter(
    df["Height"], df["Weight"], c=df["Class"], cmap="coolwarm", edgecolors="k", s=100
)
plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("KNN Classification (Athlete vs Non-Athlete)")
plt.show()
