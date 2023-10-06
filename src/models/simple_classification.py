import numpy as np
#from scipy.cluster.vq import whiten, kmeans, vq
import matplotlib.pyplot as plt
from statistics import mode
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering
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

from pathlib import Path
import tempfile

import wandb

class BaseCodeBookClassifier:
    def __init__(self,
                input_length,
                config):
        """
        Base model for classifying the codebook 
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

        full_ts_s = [] #TODO: make static. List containing index representations of ts in codebook.
        y_labels = []

        for i in range(number_of_batches):
            try:
                x, y = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(data_loader)
                x, y = next(dataloader_iterator)
            
            z_q, s = self.encode_to_z_q(x, self.encoder, self.vq_model)
            
            for i, s_i in enumerate(s):
                full_ts_s.append(s_i)
  
        full_ts_s = torch.stack(full_ts_s)
        return full_ts_s.numpy()
    

    def remap_clusters(self, true_labels, cluster_labels):
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

    def classify(data_loader):
        raise NotImplemented

class KMeansCodeBook(BaseCodeBookClassifier):
    """
    Kmeans codebook classifier
    """
    def __init__(self, 
                input_length,
                config): 
        super().__init__(input_length, config)
    
    def classify(self, data_loader):

        dataset_codebook_indicies = self.run_through_codebook(data_loader)
        y_labels = data_loader.dataset.Y.flatten().astype(int)

        k = len(np.unique(y_labels)) #number of clusters

        scaler = StandardScaler()
        scaled_full_ts_s = scaler.fit_transform(dataset_codebook_indicies)
        kmeans = KMeans(init="random", n_init=10, n_clusters=k, max_iter=300)
        kmeans.fit(scaled_full_ts_s)

        remapped_labels = self.remap_clusters(y_labels, kmeans.labels_)
        
        return{
            "sse": kmeans.inertia_, #The same as sse
            "centers": kmeans.cluster_centers_,
            "labels": remapped_labels,
            "accuracy": accuracy_score(y_labels, remapped_labels)
        }


    def multiple_classifications(self, data_loader, num = 100):
        from tqdm import tqdm

        accuracies = np.zeros(num)
        sse = np.zeros(num)

        for i in tqdm(range(num)):
            cluster_data = self.classify(data_loader)
            accuracies[i] = cluster_data['accuracy']
            sse[i] = cluster_data['sse']

        return accuracies, sse
