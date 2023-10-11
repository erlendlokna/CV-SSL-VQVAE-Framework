import numpy as np
#from scipy.cluster.vq import whiten, kmeans, vq
import matplotlib.pyplot as plt
from statistics import mode
from src.models.encoder_decoder import VQVAEEncoder, VQVAEDecoder
from src.models.vq import VectorQuantize

from src.utils import (compute_downsample_rate,
                       get_root_dir,
                        time_to_timefreq,
                        timefreq_to_time,
                        quantize,
                        freeze)

from src.models.base_model import BaseModel, detach_the_unnecessary
from supervised_FCN.example_pretrained_model_loading import load_pretrained_FCN
from torch.utils.data import DataLoader

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from pathlib import Path
import tempfile
from tqdm import tqdm

def remap_clusters(true_labels, cluster_labels):
    """
    Function for remapping labels from the zeroshot classifier to the most
    frequent labels in the true labels.
    """
    unique_clusters = np.unique(cluster_labels)
    mapping = {}
    
    for cluster in unique_clusters:
        # Find the most frequent true label in this cluster
        true_label_counts = np.bincount(true_labels[cluster_labels == cluster])
        most_frequent_true_label = np.argmax(true_label_counts)
        
        # Map cluster label to most frequent true label
        mapping[cluster] = most_frequent_true_label
    
    # Remap cluster labels
    remapped_labels = np.vectorize(lambda x: mapping[x])(cluster_labels)
    
    return remapped_labels

class BaseVQVAE:
    def __init__(self,
                input_length,
                config):
        """
        Base model for classifying the VQVAE codebook. 
        """
        self.input_length = input_length
        self.config = config
        self.n_fft = config['VQVAE']['n_fft']

        dim = config['encoder']['dim']
        in_channels = config['dataset']['in_channels']

        downsampled_width = config['encoder']['downsampled_width']
        downsampled_rate = compute_downsample_rate(input_length, self.n_fft, downsampled_width)

        self.encoder = VQVAEEncoder(dim, 2*in_channels, downsampled_rate, config['encoder']['n_resnet_blocks'])
        self.vq_model = VectorQuantize(dim, config['VQVAE']['codebook']['size'], **config['VQVAE'])
        self.decoder = VQVAEDecoder(dim, 2 * in_channels, downsampled_rate, config['decoder']['n_resnet_blocks'])
        
        #grabbing pretrained models:
        dataset_name = config['dataset']['dataset_name']
        self.load(self.encoder, get_root_dir().joinpath('saved_models'), f'encoder-{dataset_name}.ckpt')
        self.load(self.decoder, get_root_dir().joinpath('saved_models'), f'decoder-{dataset_name}.ckpt')
        self.load(self.vq_model, get_root_dir().joinpath('saved_models'), f'vq_model-{dataset_name}.ckpt')

        self.classifier_name = "Kmeans"
        
    def load(self, model, dirname, fname):
        """
        model: instance
        path_to_saved_model_fname: path to the ckpt file (i.e., trained model)
        """
        try:
            model.load_state_dict(torch.load(dirname.joinpath(fname)))
        except FileNotFoundError:
            dirname = Path(tempfile.gettempdir())
            model.load_state_dict(torch.load(dirname.joinpath(fname)))
    
    def encode_to_z_q(self, x, encoder: VQVAEEncoder, vq_model: VectorQuantize):
        """
        x: (B, C, L)
        """
        C = x.shape[1]
        xf = time_to_timefreq(x, self.n_fft, C)  # (B, C, H, W)
        
        z = encoder(xf)  # (b c h w)
        z_q, indices, vq_loss, perplexity = quantize(z, vq_model)  # (b c h w), (b (h w) h), ...
        return z_q, indices

    def run_through_codebook(self, data_loader):
        #collecting all the timeseries codebook index representations:
        dataloader_iterator = iter(data_loader)
        number_of_batches = len(data_loader)

        full_ts_zqs = [] #TODO: make static. List containing zqs for each timeseries in data_loader

        for i in range(number_of_batches):
            try:
                x, y = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(data_loader)
                x, y = next(dataloader_iterator)
            
            z_q, s = self.encode_to_z_q(x, self.encoder, self.vq_model)

            for i, zq_i in enumerate(z_q):
                
                full_ts_zqs.append(zq_i.detach().flatten().numpy())
    
        return np.array(full_ts_zqs)
    

    def predict(data_loader):
        raise NotImplemented

#Making the following classes generic for both supervised and unsupervised classification.

class KMeans_VQVAE(BaseVQVAE):
    """
    Zero Shot Kmeans VQVAE Classifier 
    """
    def __init__(self, 
                input_length,
                config, 
                train_data_loader = None,
                test_size = None):
        """
        Fits the kmeans on the train_data_loader.

        ---
        If you want to train and test on the same dataloader using a split use:
        train_data_loader and set split_train_data = True
        """
        super().__init__(input_length, config)

        if test_size:
            #Do a split on the dataloader
            zqs = self.run_through_codebook(train_data_loader)
            y_labs = train_data_loader.dataset.Y.flatten().astype(int)
            self.zqs_train, self.zqs_test, self.y_train, self.y_test = train_test_split(zqs, y_labs, test_size=test_size)

        else:
            self.zqs_train = self.run_through_codebook(train_data_loader)
            self.y_train = train_data_loader.dataset.Y.flatten().astype(int)
            self.zqs_test = None #zqs_test and y_test will be set in predict()
            self.y_test = None

        self.name = "Kmeans_VQVAE"

    
    def predict(self, test_data_loader=None):
        zqs_test, y_test = self.zqs_test, self.y_test

        if zqs_test is None and y_test is None:
            assert test_data_loader, f"Please add a data_loader or set a test_size in the {self.name} constructor"
            zqs_test = self.run_through_codebook(test_data_loader)
            y_test = test_data_loader.dataset.Y.flatten().astype(int)
        
        k = len(np.unique(self.y_train)) #number of clusters

        scaler = StandardScaler()
        scaled_zqs_train = scaler.fit_transform(self.zqs_train)
        scaled_zqs_test = scaler.fit_transform(zqs_test)

        self.kmeans = KMeans(init="random", n_init=10, n_clusters=k, max_iter=300)
        self.kmeans.fit(scaled_zqs_train) #fit

        y_pred = self.kmeans.predict(scaled_zqs_test) #predict
        y_pred = remap_clusters(y_test, y_pred) #matching labels
        
        return {'predictions': y_pred,
                'accuracy': metrics.accuracy_score(y_test, y_pred),
                'confusion_matrix': metrics.confusion_matrix(y_test, y_pred)}


class SVM_VQVAE(BaseVQVAE):
    def __init__(self, 
                input_length, config,
                train_data_loader = None,
                test_size = None,
                kernel='linear'):
        """
        Does a SVM on the discrete latent variables generated by the encoder and vector quantifier.

        ---
        If you want to train and test on the same dataloader using a split use:
        train_data_loader and set split_train_data = True
        """
        super().__init__(input_length, config)

        if test_size:
            #Do a split on the dataloader
            zqs = self.run_through_codebook(train_data_loader)
            y_labs = train_data_loader.dataset.Y.flatten().astype(int)
            self.zqs_train, self.zqs_test, self.y_train, self.y_test = train_test_split(zqs, y_labs, test_size=test_size)

        else:
            self.zqs_train = self.run_through_codebook(train_data_loader)
            self.y_train = train_data_loader.dataset.Y.flatten().astype(int)
            self.zqs_test = None #zqs_test and y_test will be set in predict()
            self.y_test = None

        self.name = "SVM_VQVAE"

        #defining and training the svm:
        self.svm = SVC(kernel=kernel)
        self.svm.fit(self.zqs_train, self.y_train)

    def predict(self, test_data_loader=None):
        zqs_test, y_test = self.zqs_test, self.y_test
        if zqs_test is None and y_test is None:
            assert test_data_loader, f"Please add a data_loader or set split_to_test_train = True in the {self.name} constructor"
            zqs_test = self.run_through_codebook(test_data_loader)
            y_test = test_data_loader.dataset.Y.flatten().astype(int)

        y_pred = self.svm.predict(zqs_test)

        return {'predictions': y_pred,
                'accuracy': metrics.accuracy_score(y_test, y_pred),
                'confusion_matrix': metrics.confusion_matrix(y_test, y_pred)}


def evaluate_classifier(model, test_data_loader=None, num_tests=50):
    accuracies = np.zeros(num_tests)

    for i in tqdm(range(num_tests)):
        prediction_data = model.predict(test_data_loader)
        accuracies[i] = prediction_data['accuracy']
    
    mean = np.mean(accuracies)
    return accuracies, mean
