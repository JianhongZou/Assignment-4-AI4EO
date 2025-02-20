#  Unsupervised Learning

The tasks in this notebook will be mainly two:

Discrimination of Sea ice and lead based on image classification based on Sentinel-2 optical data.

Discrimination of Sea ice and lead based on altimetry data classification based on Sentinel-3 altimetry data.

## Basic code
### 1 Connect to the googledrive
```python
from google.colab import drive

drive.mount('/content/drive')
```

### 2 Install basic packages
```python
pip install rasterio
pip install netCDF4
```
### K-mean plotting
```python
# Python code for K-means clustering

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

import numpy as np

  

# Sample data

X = np.random.rand(100,  2)

  

# K-means model

kmeans = KMeans(n_clusters=4)

kmeans.fit(X)

y_kmeans = kmeans.predict(X)

  

# Plotting

plt.scatter(X[:,  0], X[:,  1], c=y_kmeans, cmap='viridis')

centers = kmeans.cluster_centers_

plt.scatter(centers[:,  0], centers[:,  1], c='black', s=200, alpha=0.5)

plt.show()
```

