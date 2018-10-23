"""
Created on Tue Oct 18 2018
@author: Supratim Haldar
@Description: My implementation of K-Means Clustering 
(unsupervised learning) algorithm
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.cluster import KMeans

# =============================================================================
# Calculate the distance from each cluster centroid, and assign each training 
# data to the closest cluster centroid
# =============================================================================
def clusterAssignment(data_X, initial_centroids):
    k = initial_centroids.shape[0]
    m = data_X.shape[0]
    closestCentroidIdx = np.zeros([m, 1])
    
    #Distance matrix of size m x k
    Distance = np.zeros([m, k])
    
    #Loop through each centroid and calculate distance from X
    for j in range(k):
        Centroid = initial_centroids[j, :]
        Distance[:, j] = np.sum(((data_X - Centroid)**2), axis=1)
    print("Distance =", Distance)
    
    # Find centroid which is nearest to the X
    closestCentroidIdx = np.argmin(Distance, axis=1).reshape(m, -1)
    print("Closest Centroid Index =", closestCentroidIdx)
    return closestCentroidIdx

# =============================================================================
# Calculate mean of all training data assigned to a particular cluster,
# and move the cluster centroid to that mean value
# =============================================================================
def moveCentroid(data_X, closestCentroidIdx, k):
    NewCentroids = np.zeros([k, data_X.shape[1]])
    
    # Loop through each centroid and find mean of X assigned to that centroid
    # Set new centroid value to that mean
    for j in range(k):
        position = np.where(closestCentroidIdx == j)
        NewCentroids[j, :] = np.mean(data_X[position[0], :])
    
    print("NewCentroids =", NewCentroids)    
    return NewCentroids


# =============================================================================
# Initialize the cluster centroids
# =============================================================================
def initCentroids():
    Initial_Centroids = np.array([[4, 4], [10, 10], [18, 18]])
    print("Initial Centroids =", Initial_Centroids)
    return Initial_Centroids

# =============================================================================
# Prepare training data
# =============================================================================
def generateData(m):
    X1 = np.array([1, 2, 1, 2, 7, 8, 7, 8, 15, 16, 15, 16])
    X2 = np.array([1, 2, 2, 1, 7, 8, 8, 7, 15, 16, 16, 15])
    data_X = np.zeros([m, 2])
    data_X[:, 0] = X1
    data_X[:, 1] = X2
    print("Input data =", data_X)
    return data_X
  

# =============================================================================
# Test my implementation of k-means algorithm
# =============================================================================
def test_kMeans():
    m = 12
    data_X = generateData(m)
    plt.scatter(data_X[:, 0], data_X[:, 1])
    
    Initial_Centroids = initCentroids()
    k = Initial_Centroids.shape[0]
    plt.plot(Initial_Centroids[:, 0], Initial_Centroids[:, 1], 'rX')
    
    num_iters = 2
    for i in range(num_iters):
        closestCentroidIdx = clusterAssignment(data_X, Initial_Centroids)
        NewCentroids = moveCentroid(data_X, closestCentroidIdx, k)
        Initial_Centroids = NewCentroids
    
    closestCentroidIdx = clusterAssignment(data_X, NewCentroids)
    plt.plot(NewCentroids[:, 0], NewCentroids[:, 1], 'go')


# =============================================================================
# Test k-means algorithm with sklearn library function to compare results with
# my implementation
# =============================================================================
def test_kMeans_sklearn():
    m = 12
    data_X = generateData(m)
    plt.scatter(data_X[:, 0], data_X[:, 1])

    kmeans = KMeans(n_clusters=3).fit(data_X)
    print("KMeans Cluster Centers =", kmeans.cluster_centers_)
    print("KMeans labels =", kmeans.labels_)
    plt.plot(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 'rx')
    
def main():
    start_time = time.time()
    test_kMeans()
    test_kMeans_sklearn()
    print("Execution time in Seconds =", time.time() - start_time)

if __name__ == '__main__':
    main()

