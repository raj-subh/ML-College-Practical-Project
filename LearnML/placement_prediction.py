import pickle
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlxtend.plotting import plot_decision_regions

df = pd.read_csv("placement.csv")

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.columns)
print(df.shape)
print(df.dtypes)

# preprocessing of dataset
df = df.iloc[:, 1:]
print(df.info())

plt.scatter(df["cgpa"], df["iq"], c=df["placement"])

x = df.iloc[:, 0:2]
y = df.iloc[:, -1]

train_test_split(x, y, test_size=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=42
)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train[:5])
print(X_test[:5])
print(y_train[:5])
print(y_test[:5])

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)
print("Model is Trained")


# Model prediction
y_pred = model.predict(X_test)

# Model evaluatoin
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# View model
plot_decision_regions(X_train, y_train.values, clf=model, legend=2)

pickle.dump(model, open("placement_model.pkl", "wb"))
