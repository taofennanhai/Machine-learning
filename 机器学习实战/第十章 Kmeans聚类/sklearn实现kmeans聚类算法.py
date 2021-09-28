import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn import metrics

X, y = make_blobs(200, 2, centers=2, random_state=1, cluster_std=2)

print(y)
plt.scatter(np.transpose(X[:, 0]), np.transpose(X[:, 1]), c=y)
plt.show()

kmeans = KMeans(algorithm='', n_clusters=2)
kmeans.fit(X)

centers = kmeans.cluster_centers_
print(centers)

plt.scatter(np.transpose(X[:, 0]), np.transpose(X[:, 1]), c=y)
plt.scatter(np.transpose(centers[:, 0]), np.transpose(centers[:, 1]), c='red', s=200, alpha=0.5)
print(metrics.calinski_harabasz_score(X, y))    # 计算分数
plt.show()

