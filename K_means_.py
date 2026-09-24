import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv("E:\PythonProject2\placement_predict_50k Dataset (3)(in).csv")

# Select required columns
data = data[
    [
        "CodingTestScore",
        "AptitudeTestScore"
    ]
].dropna()

# Features for K-Means
X = data[
    [
        "CodingTestScore",
        "AptitudeTestScore"
    ]
]

print("\nSelected Features:")
print(X.head())

print("\nNumber of records after removing missing values:")
print(len(X))

# Create K-Means model
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# Apply K-Means
data["Cluster"] = kmeans.fit_predict(X)

# Display clustered data
print("\nClustered Data:")
print(
    data[
        [
            "CodingTestScore",
            "AptitudeTestScore",
            "Cluster"
        ]
    ].head(10)
)

# Display cluster centres
print("\nCluster Centres:")
print(kmeans.cluster_centers_)

# Display number of students in each cluster
print("\nNumber Of Students in Each Cluster:")
print(
    data["Cluster"]
    .value_counts()
    .sort_index()
)

# Plot clusters
plt.scatter(
    data["CodingTestScore"],
    data["AptitudeTestScore"],
    c=data["Cluster"]
)

# Plot cluster centres
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker="X",
    s=200
)

plt.xlabel("CodingTestScore")
plt.ylabel("AptitudeTestScore")
plt.title("K-Means Clustering")

plt.show()