'''from data_loader import load_data
from K_Calculation_K_Means import main as calculate_k
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from k_means_per
def perform_kmeans(X,k):
    model = KMeans(n_clusters=k, random_state=42,n_init=10)
    clusters = model.fit_predict(X)

    return model,clusters

def plot_clusters(X,clusters):

    x_index = int(input("Enter feature number for X-axis: "))
    y_index = int(input("Enter feature number for Y-axis: "))

    x_feature = X.cloumns
'''