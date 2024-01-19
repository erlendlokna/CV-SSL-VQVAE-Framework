import torch

import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import silhouette_score


import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

def test_model_representations(training_data, test_data):
    """
    training_data: tuple of ztr and ytr. Training latent representations and their respective labels
    test_data: tuple of zts and yts. Test latent representations and their respective labels
    ----
    returns: dict with representation tests results
    """
    #data preprocessing
    ztr, ytr = training_data
    zts, yts = test_data 

    ztr = torch.flatten(ztr, start_dim=1).detach().cpu().numpy()
    zts = torch.flatten(zts, start_dim=1).detach().cpu().numpy()
    ytr = torch.flatten(ytr, start_dim=0).detach().cpu().numpy()
    yts = torch.flatten(yts, start_dim=0).detach().cpu().numpy()

    ztr, zts = minmax_scale(ztr, zts) #scaling

    z = np.concatenate((ztr, zts), axis=0)
    y = np.concatenate((ytr, yts), axis=0)

    #tests:
    intristic_dim = intristic_dimension(z.reshape(-1, z.shape[-1]))
    svm_acc = svm_test(ztr, zts, ytr, yts)
    #silhuettes = kmeans_clustering_silhouette(z, y, n_runs=15)
    #sil_mean, sil_std = np.mean(silhuettes), np.std(silhuettes)
    knn1_acc, knn5_acc, knn10_acc = knn_test(ztr, zts, ytr, yts)

    return {
        'intrinstic_dim': intristic_dim,
        'svm_acc': svm_acc,
        #'sil_mean': sil_mean,
        #'sil_std': sil_std,
        #'svm_rbf': svm_gs_rbf_acc,
        'knn1_acc': knn1_acc,
        'knn5_acc': knn5_acc,
        'knn10_acc': knn10_acc,
        #'km_nmi_mean': km_nmi_mean,
        #'km_nmi_std': km_nmi_std
    }

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
    a = metrics.accuracy_score(y_true=y_test, y_pred=y_pred_svm)
    if not silent: print("SVM test finished. Accuracy: ", a)
    return a

def svm_test_gs_rbf(Z_train, Z_test, y_train, y_test, silent=False):
    if not silent:
        print("Fitting grid search SVM..")

    svc = SVC(kernel='rbf', max_iter=5000)
    parameters = {'C': [10 ** i for i in range(-4, 5)]}
    search = GridSearchCV(svc, parameters, verbose=0)
    search.fit(Z_train, y_train.ravel())

    svm_clf = search.best_estimator_

    y_pred_svm = svm_clf.predict(Z_test)

    a = metrics.accuracy_score(y_true=y_test, y_pred=y_pred_svm)
    print("svm rbf finished. Accuracy:", a)
    return a


def intristic_dimension(zqs, threshold=0.95):
    print("Calculating intrinstic dimension..")
    pca = PCA(threshold)  # Set n_components to the variance threshold
    pca.fit(zqs)
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    intrinsic_dim = np.argmax(cumulative_variance >= threshold) + 1
    print("Calculation finished.")
    return intrinsic_dim

def kmeans_clustering_silhouette(Z, Y, n_runs=10):
    silhouette_scores = []
    n_clusters = len(np.unique(Y))

    for _ in range(n_runs):
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        Y_preds = kmeans.fit_predict(Z)
        
        # Calculate silhouette score
        score = silhouette_score(Z, Y_preds)  # Corrected to use Z and Y_preds
        silhouette_scores.append(score)

    # Return the average silhouette score
    return silhouette_scores

def find_optimal_k(Z, Y, max_clusters=None, n_runs=10):
    if max_clusters is None:
        max_clusters = len(np.unique(Y))  # A starting point, adjust as necessary
    
    best_k = 2
    best_score = -1
    silhouette_avgs = []
    
    # Try different numbers of clusters
    for k in range(2, max_clusters+1):
        current_silhouette_scores = []
        for _ in range(n_runs):
            kmeans = KMeans(n_clusters=k, n_init=10)
            Y_preds = kmeans.fit_predict(Z)
            score = silhouette_score(Z, Y_preds)
            current_silhouette_scores.append(score)
        
        avg_score = np.mean(current_silhouette_scores)
        silhouette_avgs.append(avg_score)
        
        if avg_score > best_score:
            best_score = avg_score
            best_k = k
            
    return best_k, best_score, silhouette_avgs


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
    
