import torch

import umap
from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from src.experiments.classnet_training import train_ClassNet, train_CNNClassNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics


import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


def knn_test(Z_train, Z_test, y_train, y_test):
    k = len(np.unique(y_train)) #number of clusters/neightbors
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(Z_train, y_train)
    y_pred_knn = neigh.predict(Z_test)
    return metrics.accuracy_score(y_true=y_test, y_pred=y_pred_knn)

def svm_test(Z_train, Z_test, y_train, y_test):
    svm = SVC(kernel='linear')
    svm.fit(Z_train, y_train)
    y_pred_svm = svm.predict(Z_test)
    return metrics.accuracy_score(y_true=y_test, y_pred=y_pred_svm)

def classnet_test(Z_train, Z_test, y_train, y_test, CNN=False, num_epochs=200):
    classnet = train_CNNClassNet(Z_train, y_train, num_epochs) if CNN else train_ClassNet(Z_train, y_train, num_epochs)
    preds = classnet.predict(Z_test)
    return metrics.accuracy_score(y_test, preds)

def intristic_dimension(zqs):
    pca = PCA(n_components=zqs.shape[0])
    pca.fit(zqs)
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    intrinsic_dim = np.argmax(cumulative_variance >= 0.95) + 1
    return intrinsic_dim


def multiple_tests(test, Z, y, n_runs, CNN=False, num_epochs=None, concatenate=True, scale=True):
    accs = np.zeros(n_runs)

    if concatenate:
        Z = StandardScaler().fit_transform(Z) if scale and not CNN else Z
    else:
        Ztr, Zts = Z
        ytr, yts = y

        if scale and not CNN: 
            scaler = StandardScaler().fit(Ztr)
            Ztr = scaler.transform(Ztr)
            Zts = scaler.transform(Zts)

    for i in tqdm(range(n_runs)):
        if concatenate:
            Ztr, Zts, ytr, yts = train_test_split(Z, y, test_size=0.2)
        
        if num_epochs is None:
            accs[i] = test(Ztr, Zts, ytr, yts)
        else:
            accs[i] = test(Ztr, Zts, ytr, yts, CNN=CNN, num_epochs=200)
    return accs

def plot_tests(results, labels):
    y_min = np.min(results)

    f, a = plt.subplots(figsize=(3, 9))
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    a.set_title("Probe tests")
    for i in range(len(results)):
        a.plot(results[i], c=colors[i])
        mean = np.mean(results[i])
        a.plot([mean for _ in range(len(results[i]))], '--', c=colors[i], label=f"{labels[i]} - mean: {mean}")
        a.set_ylim(0.7 * y_min, 1)
    f.legend()
    plt.show()