import torch

import umap
from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics


import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


from src.utils import (
    UMAP_wrapper, PCA_wrapper, UMAP_plots
)


def supervised_test(Z_train, Z_test, y_train, y_test, embed=False) -> dict:
    """
    Runs train and fit for models: KNN, SVM, LDA

    Returns accuracies.
    """

    accuracies = {'knn': 0, 'knn_pca':None, 'knn_umap': None, #init
                'svm': 0, 'svm_pca': None, 'svm_umap': None,
                'lda': 0, 'lda_pca': None, 'lda_umap': None}

    k = len(np.unique(y_train)) #number of clusters/neightbors

    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(Z_train, y_train)
    y_pred_knn = neigh.predict(Z_test)
    
    accuracies['knn'] = metrics.accuracy_score(y_true=y_test, y_pred=y_pred_knn)

    if embed:
        pca_neigh = PCA_wrapper(KNeighborsClassifier(n_neighbors=k), var_explained_crit=0.95)
        umap_neigh = UMAP_wrapper(KNeighborsClassifier(n_neighbors=k), n_comps = 10)

        pca_neigh.fit(Z_train, y_train)
        umap_neigh.fit(Z_train, y_train)

        y_pca_neigh_pred = pca_neigh.predict(Z_test)
        y_umap_neigh_pred = umap_neigh.predict(Z_test)
        accuracies['knn_pca'] = metrics.accuracy_score(y_true=y_test, y_pred=y_pca_neigh_pred)
        accuracies['knn_umap'] = metrics.accuracy_score(y_true=y_test, y_pred=y_umap_neigh_pred)

    svm = SVC(kernel='linear')
    svm.fit(Z_train, y_train)
    y_pred_svm = svm.predict(Z_test)
    accuracies['svm'] = metrics.accuracy_score(y_true=y_test, y_pred=y_pred_svm)
    
    if embed:
        pca_svm = PCA_wrapper(SVC(kernel='linear'), var_explained_crit=0.95)
        umap_svm = UMAP_wrapper(SVC(kernel='linear'), n_comps = 20)
            
        pca_svm.fit(Z_train, y_train)   
        umap_svm.fit(Z_train, y_train)  
            
        y_pca_svm_pred = pca_svm.predict(Z_test) 
        y_umap_svm_pred = umap_svm.predict(Z_test)   
            
        accuracies['svm_pca'] = metrics.accuracy_score(y_true=y_test, y_pred=y_pca_svm_pred)
        accuracies['svm_umap'] = metrics.accuracy_score(y_true=y_test, y_pred=y_umap_svm_pred)

    clf = LinearDiscriminantAnalysis()
    clf.fit(Z_train, y_train)

    y_pred_lda = clf.predict(Z_test)
    accuracies['lda'] = metrics.accuracy_score(y_pred=y_pred_lda, y_true=y_test)

    if embed:
        pca_lda = PCA_wrapper(LinearDiscriminantAnalysis(), var_explained_crit=0.95)
        umap_lda = UMAP_wrapper(LinearDiscriminantAnalysis(), n_comps = 20)

        pca_lda.fit(Z_train, y_train)
        umap_lda.fit(Z_train, y_train)

        y_pca_lda_pred = pca_lda.predict(Z_test)
        y_umap_lda_pred = umap_lda.predict(Z_test)

        accuracies['lda_pca'] = metrics.accuracy_score(y_true=y_test, y_pred=y_pca_lda_pred)
        accuracies['lda_umap'] = metrics.accuracy_score(y_true=y_test, y_pred=y_umap_lda_pred)

    return accuracies