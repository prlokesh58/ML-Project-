import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator


# Load Dataset
data = pd.read_csv(r"E:\PythonProject2\preprocessed_placement.csv")

# Take maximum 1000 samples
if len(data) > 1000:
    data = data.sample(n=1000, random_state=42)

# Remove target column
X = data.drop("PlacementStatus", axis=1)


# Find optimal EPS
def find_eps(X, min_samples=5):

    neighbors = NearestNeighbors(n_neighbors=min_samples)

    neighbors.fit(X)

    distances, indices = neighbors.kneighbors(X)

    k_distances = sorted(distances[:, -1])

    x = list(range(len(k_distances)))

    knee_locator = KneeLocator(
        x,
        k_distances,
        curve="convex",
        direction="increasing"
    )

    if knee_locator.knee is not None:
        eps = round(k_distances[knee_locator.knee], 3)
    else:
        eps = round(k_distances[-1], 3)

    # K-Distance Graph
    plt.figure(figsize=(8, 5))

    plt.plot(k_distances)

    if knee_locator.knee is not None:
        plt.axvline(
            x=knee_locator.knee,
            linestyle="--"
        )

    plt.xlabel("Data Points")
    plt.ylabel("5th Nearest Neighbors Distance")
    plt.title("K-Distance Graph")
    plt.grid()

    plt.show()

    print("Optimal EPS:", eps)

    return eps


# Find EPS
eps = find_eps(X, min_samples=5)


# Create DBSCAN model
model = DBSCAN(
    eps=eps,
    min_samples=5
)


# Fit DBSCAN
labels = model.fit_predict(X)


# Find Core Points
core = model.core_sample_indices_


# Find Noise Points
noise = labels == -1


# Find Border Points
index = pd.Series(range(len(X)))

border = ~noise & ~index.isin(core)


# Count Clusters
clusters = len(set(labels)) - (1 if -1 in labels else 0)


# Print Results
print()
print("========== DBSCAN RESULTS ==========")
print("Clusters:", clusters)
print("Core Points:", len(core))
print("Border Points:", border.sum())
print("Noise Points:", noise.sum())


# Plot DBSCAN Clusters
plt.figure(figsize=(8, 5))

plt.scatter(
    X.iloc[:, 0],
    X.iloc[:, 1],
    c=labels
)

plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])
plt.title("DBSCAN Clustering")
plt.grid()

plt.show()