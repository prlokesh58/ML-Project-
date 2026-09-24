import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv("E:\PythonProject2\preprocessed_placement.csv")

# Select required columns
data = data[
    [
        "CGPA",
        "AptitudeTestScore"
    ]
].dropna()

# Features for K-Means
X = data[
    [
        "CGPA",
        "AptitudeTestScore"
    ]
]

print("\nSelected Features:")
print(X.head())

print("\nNumber of records after removing missing values:")
print(len(X))

# -----------------------------
# ELBOW METHOD
# -----------------------------

wcss = []

for k in range(1, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

# Plot Elbow Curve
plt.plot(
    range(1, 11),
    wcss,
    marker="o"
)

plt.xlabel("Number of clusters (k)")
plt.ylabel("WCSS")
plt.title("Elbow Method")

plt.show()


# -----------------------------
# FINAL K-MEANS
# -----------------------------

# Choose number of clusters
k = 4

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

# Assign clusters
data["Cluster"] = kmeans.fit_predict(X)

print("\nCluster Results:")
print(
    data[
        [
            "CGPA",
            "AptitudeTestScore",
            "Cluster"
        ]
    ].head(10)
)

print("\nCluster Centres:")
print(kmeans.cluster_centers_)

print("\nNumber Of Students in Each Cluster:")
print(
    data["Cluster"]
    .value_counts()
    .sort_index()
)