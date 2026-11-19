# your code here
import seaborn as sns
from time import time
from sklearn.cluster import KMeans

# Exercise 2: load diamonds and keep only the numeric columns
diamonds = sns.load_dataset('diamonds')
diamonds_numeric = diamonds.select_dtypes(include='number')


def kmeans(X, k):
    """
    Perform K-means clustering on the given data.

    Parameters:
    X (array-like): The input data to cluster.
    k (int): The number of clusters to form.

    Returns:
    tuple: A tuple containing the cluster centers and the cluster labels.
    """

    # Initialize the KMeans model
    kmeans_model = KMeans(n_clusters=k)

    # Fit the model to the data
    kmeans_model.fit(X)

    # Get the cluster centroids and labels
    centroid = kmeans_model.cluster_centers_
    labels = kmeans_model.labels_

    return centroid, labels


def kmeans_diamonds(n, k):
    """
    Run kmeans with k clusters on the first n rows of the numeric diamonds data.

    Returns:
    tuple: (centroids, labels) as returned by kmeans().
    """
    X = diamonds_numeric.iloc[:n].to_numpy()
    return kmeans(X, k)


def kmeans_timer(n, k, n_iter=5):
    """
    Run kmeans_diamonds(n, k) n_iter times and return the average runtime in seconds.
    """
    times = []
    for _ in range(n_iter):
        start = time()
        _ = kmeans_diamonds(n, k)
        time_taken = time() - start
        times.append(time_taken)
    return sum(times) / len(times)
