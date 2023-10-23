import torch

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


from src.utils import (
    UMAP_wrapper, PCA_wrapper, UMAP_plots
)

from src.experiments.supervised_tests import supervised_test
from src.experiments.unsupervised_tests import kmeans_test

def run_tests(Z, y, n_runs, embed, test_size, concatenate, scale=True):
        """
        Runs n_runs of tests on Z, y. Splitting each iteration
        """
        test_accs = {'knn': np.zeros(n_runs), 'knn_pca':np.zeros(n_runs), 'knn_umap': np.zeros(n_runs), #init
                'svm': np.zeros(n_runs), 'svm_pca': np.zeros(n_runs), 'svm_umap': np.zeros(n_runs),
                'lda': np.zeros(n_runs), 'lda_pca': np.zeros(n_runs), 'lda_umap': np.zeros(n_runs)}

        if concatenate:
            Zs = np.concatenate(Z, axis=0)
            Zs = StandardScaler().fit_transform(Zs) if scale else Zs
            ys = np.concatenate(y, axis=0)
        else:
            Ztr, Zts = Z
            ytr, yts = y

            if scale: 
                scaler = StandardScaler().fit(Ztr)
                Ztr = scaler.transform(Ztr)
                Zts = scaler.transform(Zts)

        for i in tqdm(range(n_runs)):
            if concatenate:
                Ztr, Zts, ytr, yts = train_test_split(Zs, ys, test_size=test_size)
            
            results = supervised_test(Ztr, Zts, ytr, yts, embed=embed)
            
            for key in results.keys():
                if results[key] is None: continue
                test_accs[key][i] = results[key]

        return test_accs

def single_test(Z, y, embed, test_size, scale, concatenate):
    if concatenate:
        Zs = np.concatenate(Z, axis=0)
        Zs = StandardScaler().fit_transform(Zs) if scale else Zs
        ys = np.concatenate(y, axis=0)
        Ztr, Zts, ytr, yts = train_test_split(Z, y, test_size)
    else:
        Ztr, Zts = Z
        ytr, yts = y
    return supervised_test(Ztr, Zts, ytr, yts)



class RepTester:
    #Tester class for VQVAE's zqs. 
    def __init__(self, VQVAE,
                 train_data_loader, test_data_loader, 
                 concatenate_zqs = True):
        self.VQVAE = VQVAE
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.concatenate = concatenate_zqs

    # ---- Preprocessing VQVAE helper functions -----
    def get_y(self): 
        ytr = self.train_data_loader.dataset.Y.flatten().astype(int)
        yts = self.test_data_loader.dataset.Y.flatten().astype(int)
        return ytr, yts

    def max_pooled_zqs(self, kernel, stride, conc=True):
        zqs_train = self.VQVAE.get_max_pooled_zqs(self.train_data_loader, kernel_size=kernel, stride=stride)
        zqs_train = torch.flatten(zqs_train, start_dim = 1).numpy()
        zqs_test = self.VQVAE.get_max_pooled_zqs(self.test_data_loader, kernel_size=kernel, stride=stride)
        zqs_test = torch.flatten(zqs_test, start_dim = 1).numpy()
        return zqs_train, zqs_test
    
    def avg_pooled_zqs(self, kernel, stride, conc=True):
        zqs_train = self.VQVAE.get_avg_pooled_zqs(self.train_data_loader, kernel_size=kernel, stride=stride)
        zqs_train = torch.flatten(zqs_train, start_dim = 1).numpy()
        zqs_test = self.VQVAE.get_avg_pooled_zqs(self.test_data_loader, kernel_size=kernel, stride=stride)
        zqs_test = torch.flatten(zqs_test, start_dim = 1).numpy()
        return zqs_train, zqs_test
    
    def global_avg_pooled_zqs(self, conc=True):
        zqs_train = self.VQVAE.get_global_avg_pooled_zqs(self.train_data_loader)
        zqs_test = self.VQVAE.get_global_avg_pooled_zqs(self.test_data_loader)
        return zqs_train, zqs_test

    def global_max_pooled_zqs(self):
        zqs_train = self.VQVAE.get_global_max_pooled_zqs(self.train_data_loader)
        zqs_test = self.VQVAE.get_global_max_pooled_zqs(self.test_data_loader)
        return zqs_train, zqs_test

    def flatten_zqs(self):
        zqs_train, _ = self.VQVAE.get_flatten_zqs_s(self.train_data_loader)
        zqs_test, _ = self.VQVAE.get_flatten_zqs_s(self.test_data_loader)
        return zqs_train, zqs_test
    
    def conv2d_zqs(self, in_channels, out_channels, kernel_size, stride, padding):
        zqs_train = self.VQVAE.get_conv2d_zqs(self.train_data_loader, in_channels, out_channels, kernel_size, stride, padding).numpy()
        zqs_test = self.VQVAE.get_conv2d_zqs(self.test_data_loader, in_channels, out_channels, kernel_size, stride, padding).numpy()
        return zqs_train, zqs_test
    
    def zqs_indicies(self):
        _, s_train = self.VQVAE.get_flatten_zqs_s(self.train_data_loader)
        _, s_test = self.VQVAE.get_flatten_zqs_s(self.test)
        return s_train, s_test
    
    # ---- Tests -----
    def test_flatten(self, n_runs = None, embed=False, scale=True, test_size = 0.2):
        """
        Runs n_runs of supervised_tests on flatten zqs.
        """
        y = self.get_y()
        Z = self.flatten_zqs()
        if n_runs:
            return run_tests(Z, y, n_runs, embed=embed, test_size=test_size, scale=scale, concatenate=self.concatenate)
        else:
            return single_test(Z, y, embed=embed, test_size=test_size, scale=scale, concatenate=self.concatenate)

    def test_max_pooling(self, n_runs = 1, kernel=2, stride=2, embed=False, scale=True, test_size = 0.2):
        """
        Runs n_runs of supervised_tests on max pooled zqs.
        """
        y = self.get_y()
        Z = self.max_pooled_zqs(kernel, stride)
        if n_runs:
            return run_tests(Z, y, n_runs, embed=embed, test_size=test_size, scale=scale, concatenate=self.concatenate)
        else:
            return single_test(Z, y,embed=embed, test_size=test_size, scale=scale, concatenate=self.concatenate)
    
    def test_avg_pooling(self, n_runs = 1, kernel=2, stride=2, embed=False, scale=True, test_size = 0.2):
        """
        Runs n_runs of supervised_tests on avg pooled zqs.
        """
        y = self.get_y()
        Z = self.avg_pooled_zqs(kernel, stride)
        if n_runs:
            return run_tests(Z, y, n_runs, embed=embed, test_size=test_size, scale=scale, concatenate=self.concatenate)
        else:
            return single_test(Z, y, embed=embed, test_size=test_size, scale=scale, concatenate=self.concatenate)
        
    def test_global_max_pool(self, n_runs, embed = False, scale=True, test_size = 0.2):
        """
        Runs n_runs of supervised tests on global max pool zqs.
        """
        y = self.get_y()
        Z = self.global_max_pooled_zqs()
        if n_runs:
            return run_tests(Z, y, n_runs, embed=embed, test_size=test_size, scale=scale, concatenate=self.concatenate)
        else:
            return single_test(Z, y, embed=embed, test_size=test_size, scale=scale, concatenate=self.concatenate)
    
    def test_global_avg_pool(self, n_runs, embed = False, scale=True, test_size = 0.2):
        """
        Runs n_runs of supervised tests on global avg zqs.
        """
        y = self.get_y()
        Z = self.global_avg_pooled_zqs()
        if n_runs:
            return run_tests(Z, y, n_runs, embed=embed, test_size=test_size, scale=scale, concatenate=self.concatenate)
        else:
            return single_test(Z, y, embed=embed, test_size=test_size, scale=scale, concatenate=self.concatenate)

    def test_conv2d(self, n_runs, embed=False, scale=True, test_size=0.2,
                    in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=1):
        """
        Runs n_runs of supervised tests on conv2d + Relu zqs.
        """
        y = self.get_y()
        Z = self.conv2d_zqs(in_channels, out_channels, kernel_size, stride, padding)
        if n_runs:
            return run_tests(Z, y, n_runs, embed=embed, test_size=test_size, scale=scale, concatenate=self.concatenate)
        else:
            return single_test(Z, y, embed=embed, test_size=test_size, scale=scale, concatenate=self.concatenate)
    

def plot_results(results, title="", embed=False):
    """
    Plotter function for the 
    """
    plt.style.use("fivethirtyeight")
    if embed:
        f, ax = plt.subplots(3, 3, figsize=(20, 20))
        i, j = 0, 0
        for k in results.keys():
            if (i) % 3 == 0 and i != 0: j+=1; i = 0
            ax[j][i].set_ylim(0, 1)
            ax[j][i].plot(results[k]); ax[j][i].set_title(k)
            mean = np.mean(results[k])
            ax[j][i].plot([mean for _ in range(len(results[k]))], '--', c='grey')
            i+=1
    else:
        f, ax = plt.subplots(1, 3, figsize=(20, 10))
        i = 0
        for k, i in results.keys():
            if all([v == 0 for v in results[k]]): continue
            ax[i].set_ylim(0, 1)
            ax[i].plot(results[k]); ax[i].set_title(k)
            mean = np.mean(results[k])
            ax[i].plot([mean for _ in range(len(results[k]))], '--', c='grey', label=f"{k} mean: {round(mean, 4)}")
            i+=1
    f.legend()
    f.suptitle(title, fontsize=16)
    plt.show()
        
    
def plot_multiple_results(results, labels, title="", embed=False):
    """
    Plotter function for the results with alpha transparency.
    """
    plt.style.use("fivethirtyeight")

    f, ax = plt.subplots(1,3, figsize=(20,20))

    for i in range(len(results)):
        res = results[i]
        j = 0
        for k in res.keys():
            if all([v == 0 for v in res[k]]): continue
            mean = np.mean(res[k])
            ax[j].set_ylim(0, 1)
            ax[j].plot(res[k], label=f"{labels[i],k}, mean: {round(mean, 4)}")
            ax[j].set_title(k)
            ax[j].plot([mean for _ in range(len(res[k]))], '--', c='grey')
            j+=1
    f.legend()
    plt.show()