import numpy as np
from scipy.cluster.vq import whiten, kmeans, vq
import matplotlib.pyplot as plt
from statistics import mode
from sklearn.metrics import accuracy_score

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

class KMeansCodeBook:
    def __init__(self,
                input_length,
                k: int,
                config, **kwargs):

        self.input_length = input_length
        self.config = config
        self.n_fft = config['VQVAE']['n_fft']
        self.k = k

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

    def classify(self, data_loader):

        #collecting all the timeseries codebook index representations:
        dataloader_iterator = iter(data_loader)
        number_of_batches = len(data_loader)

        full_ts_s = [] #TODO: make static
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
                y_labels.append(y[i])
                
        full_ts_s = torch.stack(full_ts_s)
        y_labels = torch.flatten(torch.stack(y_labels))

        #Kmeans:
        full_ts_s_normalized = whiten(full_ts_s) #normalising         
        centroids, mean_dist = kmeans(full_ts_s_normalized, self.k)
        clusters, dist = vq(full_ts_s_normalized, centroids)

        clusters = self.filter_cluster_labels(clusters, y_labels)
        
        print(accuracy_score(clusters, y_labels))


    def filter_cluster_labels(self, clusters, ylabels):
        #print(clusters.shape)
        #print(ylabels.shape)



        mode_0 = mode(clusters[np.where(ylabels == 0)])
        clusters[np.where(clusters == mode_0)] = 0

        mode_1 = mode(clusters[np.where(ylabels == 1)])
        clusters[np.where(clusters == mode_1)] = 1

        mode_2 = mode(clusters[np.where(ylabels == 2)])
        clusters[np.where(clusters == mode_1)] = 2
        
        return clusters


        #(np.where(ylabels == 0))

        #print(clusters)
        #print(ylabels)
