import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def load_data():

    data = pd.read_csv(r"E:\PythonProject2\preprocessed_placement.csv")

    return data


def manual_k():

    print("\n-- Manual K Selection --")

    k = int(input("Enter the number of clusters (K-Value): "))

    while k < 2:
        print("\n-- K must be greater than or equal to 2 --")
        k = int(input("Enter the number of clusters (K-Value): "))

    return k


def elbow_method():

    X = load_data()

    k_values = range(2, 11)
    wcss = []

    for k in k_values:

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        kmeans.fit(X)

        wcss.append(kmeans.inertia_)

    print("\nWCSS Values:")

    for k, value in zip(k_values, wcss):
        print("K =", k, "WCSS =", value)

    plt.plot(k_values, wcss, marker='o')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("WCSS")
    plt.title("Elbow Method")
    plt.xticks(k_values)
    plt.grid(True)
    plt.show()

    k = int(input("\nEnter the number of clusters (K-Value): "))

    return k


def silhouette_method():

    X = load_data()

    k_values = range(2, 11)
    scores = []

    sample_size = min(5000, len(X))

    for k in k_values:

        print("Calculating silhouette score for K =", k)

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = kmeans.fit_predict(X)

        score = silhouette_score(
            X,
            labels,
            sample_size=sample_size,
            random_state=42
        )

        scores.append(score)

        print("Silhouette score =", score)

    best_index = scores.index(max(scores))
    best_k = list(k_values)[best_index]

    print("\nHighest silhouette score =", max(scores))
    print("Best K =", best_k)

    plt.plot(k_values, scores, marker='o')

    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Method")
    plt.xticks(k_values)
    plt.grid(True)
    plt.show()

    return best_k


def select_method():

    print("\n-- Cluster Selection Methods --")
    print("1. Manual K Selection")
    print("2. Elbow Method")
    print("3. Silhouette Method")

    choice = int(input("\nEnter your choice: "))

    return choice


def main():

    choice = select_method()

    if choice == 1:

        k = manual_k()

    elif choice == 2:

        k = elbow_method()

    elif choice == 3:

        k = silhouette_method()

    else:

        print("Invalid choice!")
        return

    print("\nSelected K =", k)

    # Load data
    X = load_data()

    # Apply K-Means
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X)

    # Add cluster labels to data
    X["Cluster"] = labels

    print("\nClustered Data:")
    print(X.head())

    print("\nCluster Counts:")
    print(X["Cluster"].value_counts().sort_index())


if __name__ == "__main__":
    main()