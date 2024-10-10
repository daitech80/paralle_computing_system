import pandas as pd
import numpy as np
from dask import array as da
from dask_ml.cluster import KMeans
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

import time

# Load the csv data file
data = pd.read_csv('sample.csv')

# select two clomun  sepal_length and sepal_width for clustering
X = data[['sepal_length', 'sepal_width']].values 
# Convert the data to Dask array for parallel processing 
X_dask = da.from_array(X, chunks=(151, X.shape[1]))  # 151 row  and x as column

# Serial KMeans implementation  the model train
st_time = time.time() # start time
kmeans_serial = KMeans(n_clusters=2, random_state=42)  # the n_cluster is 2 and random_state 42
kmeans_serial.fit(X)
seq_time = time.time() - st_time # end time

# Parallel KMeans implementation using Dask model train
st_time = time.time() # start time 
kmeans_parallel = KMeans(n_clusters=2,  random_state=42)
kmeans_parallel.fit(X_dask)
parallel_time = time.time() - st_time # end time


#  compare the cluster centers
print("the cluster centers of serial:", kmeans_serial.cluster_centers_)
print("the cluster centers of Parallel:", kmeans_parallel.cluster_centers_)

# Create new synthetic data for prediction
# x_addis = data[['sepal_length', 'sepal_width']].sample(n=10).values
x_addis, _ = make_blobs(n_samples=10, n_features=X.shape[1], centers=2, random_state=42)

# Predict using the serial model
st_time = time.time()  # start time 
x_addis_dask = da.from_array(x_addis, chunks=(115, X.shape[1]))
predict_serial = kmeans_serial.predict(x_addis)
predict_time_serial = time.time() - st_time  # end time of prediction

# Predict using the parallel model
st_time = time.time()  
x_addis_dask = da.from_array(x_addis, chunks=(115, X.shape[1]))  # Convert new data to Dask array(da)
predictions_parallel = kmeans_parallel.predict(x_addis) 
predict_time_parallel = time.time() - st_time  

# Output the results of training and prediction
print(f"serial Kmeans token Time: {seq_time:.2f} seconds")
print(f"Parallel Kmeans token Time: {parallel_time:.2f} seconds")
print("Predictions serial:", predict_serial)
print(f"Prediction time serial: {predict_time_serial:.4f} seconds") 
print("Predictions in Parallel:", predictions_parallel)
print(f"Prediction in paralle: {predict_time_parallel:.4f} seconds")   
