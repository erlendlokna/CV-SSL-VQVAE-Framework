import torch

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from src.models.vqvae_representations import BaseVQVAE

from src.utils import (
    UMAP_wrapper, PCA_wrapper, UMAP_plots
)

from src.experiments.supervised_tests import supervised_test
from src.experiments.unsupervised_tests import kmeans_test

def run_tests(Z, y, n_runs, embed, test_size):
        """
        Runs n_runs of tests on Z, y. Splitting each iteration
        """
        test_accs = {'knn': np.zeros(n_runs), 'knn_pca':np.zeros(n_runs), 'knn_umap': np.zeros(n_runs), #init
                'svm': np.zeros(n_runs), 'svm_pca': np.zeros(n_runs), 'svm_umap': np.zeros(n_runs),
                'lda': np.zeros(n_runs), 'lda_pca': np.zeros(n_runs), 'lda_umap': np.zeros(n_runs)}

        for i in tqdm(range(n_runs)):
            Ztr, Zts, ytr, yts = train_test_split(Z, y, test_size=test_size)
            results = supervised_test(Ztr, Zts, ytr, yts, embed=embed)
            
            for key in results.keys():
                if results[key] is None: continue
                test_accs[key][i] = results[key]

        return test_accs

def single_test(Z, y, embed, test_size):
    Ztr, Zts, ytr, yts = train_test_split(Z, y)
    return supervised_test(Ztr, Zts, ytr, yts)



class RepTester(BaseVQVAE):
    #Tester class for VQVAE's zqs. 
    def __init__(self, config, input_length,
                 train_data_loader, test_data_loader):
        super().__init__(input_length, config) #initialising the VQVAE model
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader

    # ---- Preprocessing helper functions -----
    def get_y(self): 
        return np.concatenate((
            self.test_data_loader.dataset.Y.flatten().astype(int),
            self.train_data_loader.dataset.Y.flatten().astype(int)), axis = 0)
    
    def max_pooled_zqs(self, kernel, stride):
        zqs_train = self.get_max_pooled_zqs(self.train_data_loader, kernel_size=kernel, stride=stride)
        zqs_train = torch.flatten(zqs_train, start_dim = 1).numpy()
        zqs_test = self.get_max_pooled_zqs(self.test_data_loader, kernel_size=kernel, stride=stride)
        zqs_test = torch.flatten(zqs_test, start_dim = 1).numpy()
        return np.concatenate((zqs_test, zqs_train), axis=0)
    
    def avg_pooled_zqs(self, kernel, stride):
        zqs_train = self.get_avg_pooled_zqs(self.train_data_loader, kernel_size=kernel, stride=stride)
        zqs_train = torch.flatten(zqs_train, start_dim = 1).numpy()
        zqs_test = self.get_avg_pooled_zqs(self.test_data_loader, kernel_size=kernel, stride=stride)
        zqs_test = torch.flatten(zqs_test, start_dim = 1).numpy()
        return np.concatenate((zqs_test, zqs_train), axis=0)
    
    def global_avg_pooled_zqs(self):
        zqs_train = self.get_global_avg_pooled_zqs(self.train_data_loader)
        zqs_test = self.get_global_avg_pooled_zqs(self.test_data_loader)
        return np.concatenate((zqs_test, zqs_train), axis=0)

    def global_max_pooled_zqs(self):
        zqs_train = self.get_global_max_pooled_zqs(self.train_data_loader)
        zqs_test = self.get_global_max_pooled_zqs(self.test_data_loader)
        return np.concatenate((zqs_test, zqs_train), axis=0)

    def flatten_zqs(self):
        zqs_train, _ = self.get_flatten_zqs_s(self.train_data_loader)
        zqs_test, _ = self.get_flatten_zqs_s(self.test_data_loader)
        return np.concatenate((zqs_test, zqs_train), axis=0)
    # ---- Tests -----
    
    def test_flatten(self, n_runs = None, embed=False, scale=True, test_size = 0.2):
        """
        Runs n_runs of supervised_tests on flatten zqs.
        """
        y = self.get_y()
        Z = self.flatten_zqs()
        if scale: Z = StandardScaler().fit_transform(Z)
        if n_runs:
            return run_tests(Z, y, n_runs, embed, test_size)
        else:
            return single_test(Z, y, embed, test_size)

    def test_max_pooling(self, n_runs = 1, kernel=2, stride=2, embed=False, scale=True, test_size = 0.2):
        """
        Runs n_runs of supervised_tests on max pooled zqs.
        """
        y = self.get_y()
        Z = self.max_pooled_zqs(kernel, stride)
        if scale: Z = StandardScaler().fit_transform(Z)
        if n_runs:
            return run_tests(Z, y, n_runs, embed, test_size)
        else:
            return single_test(Z, y, embed, test_size)
    
    def test_avg_pooling(self, n_runs = 1, kernel=2, stride=2, embed=False, scale=True, test_size = 0.2):
        """
        Runs n_runs of supervised_tests on avg pooled zqs.
        """
        y = self.get_y()
        Z = self.avg_pooled_zqs(kernel, stride)
        if scale: Z = StandardScaler().fit_transform(Z)
        if n_runs:
            return run_tests(Z, y, n_runs, embed, test_size)
        else:
            return single_test(Z, y, embed, test_size)
    
    def test_global_max_pool(self, n_runs, embed = False, scale=True, test_size = 0.2):
        """
        Runs n_runs of supervised tests on global max pool zqs.
        """
        y = self.get_y()
        Z = self.global_max_pooled_zqs()
        if scale: Z = StandardScaler().fit_transform(Z)
        if n_runs:
            return run_tests(Z, y, n_runs, embed, test_size)
        else:
            return single_test(Z, y, embed, test_size)
    
    def test_global_avg_pool(self, n_runs, embed = False, scale=True, test_size = 0.2):
        """
        Runs n_runs of supervised tests on global avg zqs.
        """
        y = self.get_y()
        Z = self.global_avg_pooled_zqs()
        if scale: Z = StandardScaler().fit_transform(Z)
        if n_runs:
            return run_tests(Z, y, n_runs, embed, test_size)
        else:
            return single_test(Z, y, embed, test_size)


def plot_results(results, embed=False):
    plt.style.use("fivethirtyeight")
    if embed:
        f, ax = plt.subplots(3, 3, figsize=(20, 20))
        i, j = 0, 0
        for k in results.keys():
            if (i) % 3 == 0 and i != 0: j+=1; i = 0
            ax[j][i].plot(results[k]); ax[j][i].set_title(k)
            mean = np.mean(results[k])
            ax[j][i].plot([mean for _ in range(len(results[k]))], '--', c='grey')
            i+=1
    else:
        f, ax = plt.subplots(1, 3, figsize=(20, 10))
        i = 0
        for k in results.keys():
            if all([v == 0 for v in results[k]]): continue
            ax[i].plot(results[k]); ax[i].set_title(k)
            mean = np.mean(results[k])
            ax[i].plot([mean for _ in range(len(results[k]))], '--', c='grey', label=f"{k} mean: {round(mean, 4)}")
            i+=1
    f.legend()
    plt.show()
        
    
    