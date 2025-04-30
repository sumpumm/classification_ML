from models.unsupervised_learning.k_means import KMeans
from sklearn import datasets
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for saving to file
import matplotlib.pyplot as plt


def main():
    # Load the dataset
    X, y = datasets.make_blobs()

    # Cluster the data using K-Means
    clf = KMeans(k=3)
    y_pred = clf.predict(X)

    # Visualize the results
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
    plt.title("K-Means Clustering Results")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig('kmeans_plot.png')  # Save plot to file
    print("Plot has been saved as 'kmeans_plot.png'")


if __name__ == "__main__":
    main()