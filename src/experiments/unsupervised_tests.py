from sklearn.cluster import KMeans
import numpy as np
from src.utils import unsupervised_score

def kmeans_test(Z, y):
    print("Running unsupervised test..")
    k = len(np.unique(y))

    kmeans = KMeans(init="random", n_init=10, n_clusters=k, max_iter=300) 
    y_pred = kmeans.fit_predict(Z)

    unsupervised_score(y, y_pred)
