# Importing pandas for data manipulation and analysis
import pandas as pd

# Importing Matplotlib for creating static, animated, and interactive visualizations
import matplotlib.pyplot as plt

# Importing Seaborn for enhanced data visualization with statistical graphs
import seaborn as sns

# Importing NumPy for numerical computations and working with arrays
import numpy as np

# Importing KMeans from scikit-learn for performing K-Means clustering
from sklearn.cluster import KMeans


# Load the dataset
df = pd.read_csv("Mall_Customers.csv")

# Display first few rows
print("✅ Dataset Loaded Successfully!")
print(df.head())

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Dataset Info
print("\nDataset Info:")
print(df.info())

# Summary Statistics
print("\nSummary Statistics:")
print(df.describe())

# Set Seaborn style
sns.set(style="whitegrid")

# Distribution of Age
plt.figure(figsize=(6, 4))
sns.histplot(df["Age"], bins=20, kde=True, color="blue")
plt.title("Age Distribution")
plt.show()

# Distribution of Annual Income
plt.figure(figsize=(6, 4))
sns.histplot(df["Annual Income (k$)"], bins=20, kde=True, color="green")
plt.title("Annual Income Distribution")
plt.show()

# Distribution of Spending Score
plt.figure(figsize=(6, 4))
sns.histplot(df["Spending Score (1-100)"], bins=20, kde=True, color="red")
plt.title("Spending Score Distribution")
plt.show()

# Relationship Between Income & Spending Score
plt.figure(figsize=(7, 5))
sns.scatterplot(x="Annual Income (k$)", y="Spending Score (1-100)", data=df, hue=df["Genre"])
plt.title("Income vs Spending Score")
plt.show()

# Boxplot to Detect Outliers
plt.figure(figsize=(6, 4))
sns.boxplot(y=df["Annual Income (k$)"], color="cyan")
plt.title("Boxplot of Annual Income")
plt.show()

# Selecting only 'Annual Income' and 'Spending Score' for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Finding the optimal K using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
plt.title("Elbow Method for Optimal K")
plt.show()

# Applying K-Means with K=5
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Visualizing the Clusters
plt.figure(figsize=(8, 5))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=df['Cluster'], cmap='viridis', s=50, alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', label="Centroids", marker='X')
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segments")
plt.legend()
plt.show()
