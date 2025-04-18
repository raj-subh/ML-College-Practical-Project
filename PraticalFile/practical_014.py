import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Step 1: Load dataset
df = pd.read_csv("D:/ML_Practice/Data/marketing_campaign_cleaned.csv")
df.columns = df.columns.str.strip()  # Just to be safe

# Step 2: Spending features
spending_cols = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
]

X = df[spending_cols]

# Step 3: Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Step 5: PCA Visualization
pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_scaled)
df["PCA1"] = pca_components[:, 0]
df["PCA2"] = pca_components[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette="Set2", s=70)
plt.title("K-Means Clusters of Customers Based on Spending")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.grid(True)
plt.show()

# Step 6: Cluster Insights
cluster_summary = df.groupby("Cluster")[spending_cols].mean()
print("\n📊 Average Spending per Cluster:\n")
print(cluster_summary)
