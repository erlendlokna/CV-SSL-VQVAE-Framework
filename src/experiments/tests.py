import torch

import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from src.experiments.classnet_training import train_ClassNet, train_CNNClassNet
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn import metrics


import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

def standard_scale(Z_train, Z_test):
    scaler = StandardScaler().fit(Z_train)
    Z_train = scaler.transform(Z_train)
    Z_test = scaler.transform(Z_test)
    return Z_train, Z_test

def minmax_scale(Z_train, Z_test):
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(Z_train)
    Z_train = scaler.transform(Z_train)
    Z_test = scaler.transform(Z_test)
    return Z_train, Z_test

def knn_test(Z_train, Z_test, y_train, y_test, silent=False):#len(np.unique(y_train)) #number of clusters/neightbors
    if not silent: print("Fitting KNNs..")
    neigh1 = KNeighborsClassifier(n_neighbors=1)
    neigh5 = KNeighborsClassifier(n_neighbors=5)
    neigh10 = KNeighborsClassifier(n_neighbors=10)
    neigh1.fit(Z_train, y_train)
    neigh5.fit(Z_train, y_train)
    neigh10.fit(Z_train, y_train)

    if not silent: print("Predicting..")
    y_pred_knn1 = neigh1.predict(Z_test)
    y_pred_knn5 = neigh5.predict(Z_test)
    y_pred_knn10 = neigh10.predict(Z_test)

    if not silent: print("KNN test finished.")
    acc1 =  metrics.accuracy_score(y_true=y_test, y_pred=y_pred_knn1)
    acc5 =  metrics.accuracy_score(y_true=y_test, y_pred=y_pred_knn5)
    acc10 =  metrics.accuracy_score(y_true=y_test, y_pred=y_pred_knn10)
    return acc1, acc5, acc10

def svm_test(Z_train, Z_test, y_train, y_test, silent=False):
    if not silent: print("Fitting SVM..")
    svm = SVC(kernel='linear')
    svm.fit(Z_train, y_train)
    if not silent: print("Predicting using SVM..")
    y_pred_svm = svm.predict(Z_test)
    if not silent: print("SVM test finished.")
    return metrics.accuracy_score(y_true=y_test, y_pred=y_pred_svm)

def svm_test_gs_rbf(Z_train, Z_test, y_train, y_test, silent=False):
    if not silent:
        print("Fitting grid search SVM..")

    svc = SVC(kernel='rbf', max_iter=5000)
    parameters = {'C': [10 ** i for i in range(-4, 5)]}
    search = GridSearchCV(svc, parameters, verbose=0)
    search.fit(Z_train, y_train.ravel())

    svm_clf = search.best_estimator_

    y_pred_svm = svm_clf.predict(Z_test)

    return metrics.accuracy_score(y_true=y_test, y_pred=y_pred_svm)

def classnet_test(Z_train, Z_test, y_train, y_test, CNN=False, num_epochs=200):
    classnet = train_CNNClassNet(Z_train, y_train, num_epochs) if CNN else train_ClassNet(Z_train, y_train, num_epochs)
    preds = classnet.predict(Z_test)
    return metrics.accuracy_score(y_test, preds)

def intristic_dimension(zqs, threshold=0.95):
    print("Calculating intrinstic dimension..")
    pca = PCA(threshold)  # Set n_components to the variance threshold
    pca.fit(zqs)
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    intrinsic_dim = np.argmax(cumulative_variance >= threshold) + 1
    print("Calculation finished.")
    return intrinsic_dim

def kmeans_clustering_test(z_train, y_train, z_test, y_test, n_runs=10):
    nmis = []
    n_clusters = len(np.unique(y_train))

    for _ in range(n_runs):
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        kmeans.fit(z_train)
        y_test_pred = kmeans.predict(z_test)
        
        # Calculate NMI
        nmi = normalized_mutual_info_score(y_test, y_test_pred, average_method='arithmetic')
        nmis.append(nmi)

    # Return the average NMI
    return np.mean(nmis), np.std(nmis)

def calculate_entropy(indices, vq_model):
    # Flatten the indices to get a 1D array of all index occurrences
    indices_flat = indices.view(-1).detach()

    # Calculate the frequency of each index
    counts = torch.bincount(indices_flat, minlength=vq_model.codebook.size(0))

    # Convert counts to probabilities
    probabilities = counts.float() / indices_flat.size(0)

    # Avoid division by zero in case some probabilities are zero
    probabilities = probabilities[probabilities > 0]

    # Calculate the entropy
    entropy = -torch.sum(probabilities * torch.log(probabilities))

    return entropy


def multiple_tests(test, Z, Y, n_runs, CNN=False, num_epochs=None, scale=True):
    accs = np.zeros(n_runs)
    concatenate = len(Z) != 2

    if concatenate:
        Z = StandardScaler().fit_transform(Z) if scale and not CNN else Z
    else:
        Ztr, Zts = Z
        ytr, yts = Y

        if scale and not CNN: 
            scaler = StandardScaler().fit(Ztr)
            Ztr = scaler.transform(Ztr)
            Zts = scaler.transform(Zts)

    for i in tqdm(range(n_runs), disable=(n_runs==1)):
        if concatenate:
            Ztr, Zts, ytr, yts = train_test_split(Z, Y, test_size=0.2)
        else:
            #shuffling
            permutation = np.random.permutation(Ztr.shape[0])
            Ztr = Ztr[permutation]
            ytr = ytr[permutation]
            
        if num_epochs is None:
            accs[i] = test(Ztr, Zts, ytr, yts, silent=(n_runs!=1))
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

def pca_plots(zqs, y):
    pca = PCA(n_components=2)
    embs = pca.fit_transform(zqs)
    f, a = plt.subplots()
    a.scatter(embs[:, 0], embs[:, 1], c=y)
    a.set_title("PCA plot")
    plt.show()

def umap_plots(zqs, y):
    embs = umap.UMAP(densmap=True).fit_transform(zqs)
    f, a = plt.subplots()
    a.scatter(embs[:, 0], embs[:, 1], c=y)
    a.set_title("UMAP plot")
    plt.show()

def tsne_plot(zqs, y):
    embs = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(zqs)
    f, a = plt.subplots()
    a.scatter(embs[:, 0], embs[:, 1], c=y)
    a.set_title("TSNE plot")
    plt.show()
    
