import torch
import numpy as np




class LatentDataHandler:
    #Tester class for VQVAE's zqs. 
    def __init__(self, VQVAE,
                 train_data_loader, test_data_loader, 
                 concatenate_zqs = True):
        self.VQVAE = VQVAE
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.concatenate = concatenate_zqs

    # ---- Preprocessing VQVAE helper functions and poolings -----
    def get_y(self): 
        ytr = self.train_data_loader.dataset.Y.flatten().astype(int)
        yts = self.test_data_loader.dataset.Y.flatten().astype(int)
        if self.concatenate: return np.concatenate((ytr, yts), axis=0)
        return ytr, yts

    def zqs(self, flatten = True):
        if flatten:
            zqs_train, _ = self.VQVAE.get_flatten_zqs_s(self.train_data_loader)
            zqs_test, _ = self.VQVAE.get_flatten_zqs_s(self.test_data_loader)
        else:
            zqs_train, _ = self.VQVAE.run_through_encoder_codebook(self.train_data_loader)
            zqs_test, _ = self.VQVAE.run_through_encoder_codebook(self.test_data_loader)

        if self.concatenate: return np.concatenate((zqs_train, zqs_test), axis=0)
        return zqs_train, zqs_test

    def max_pooled_zqs(self, kernel, stride):
        zqs_train = self.VQVAE.get_max_pooled_zqs(self.train_data_loader, kernel_size=kernel, stride=stride)
        zqs_train = torch.flatten(zqs_train, start_dim = 1).numpy()
        zqs_test = self.VQVAE.get_max_pooled_zqs(self.test_data_loader, kernel_size=kernel, stride=stride)
        zqs_test = torch.flatten(zqs_test, start_dim = 1).numpy()
        if self.concatenate: return np.concatenate((zqs_train, zqs_test), axis=0)
        return zqs_train, zqs_test
    
    def avg_pooled_zqs(self, kernel, stride):
        zqs_train = self.VQVAE.get_avg_pooled_zqs(self.train_data_loader, kernel_size=kernel, stride=stride)
        zqs_train = torch.flatten(zqs_train, start_dim = 1).numpy()
        zqs_test = self.VQVAE.get_avg_pooled_zqs(self.test_data_loader, kernel_size=kernel, stride=stride)
        zqs_test = torch.flatten(zqs_test, start_dim = 1).numpy()
        if self.concatenate: return np.concatenate((zqs_train, zqs_test), axis=0)
        return zqs_train, zqs_test
    
    def global_avg_pooled_zqs(self):
        zqs_train = self.VQVAE.get_global_avg_pooled_zqs(self.train_data_loader)
        zqs_test = self.VQVAE.get_global_avg_pooled_zqs(self.test_data_loader)
        if self.concatenate: return np.concatenate((zqs_train, zqs_test), axis=0)
        return zqs_train, zqs_test

    def global_max_pooled_zqs(self):
        zqs_train = self.VQVAE.get_global_max_pooled_zqs(self.train_data_loader)
        zqs_test = self.VQVAE.get_global_max_pooled_zqs(self.test_data_loader)
        if self.concatenate: return np.concatenate((zqs_train, zqs_test), axis=0)
        return zqs_train, zqs_test
    
    def conv2d_zqs(self, in_channels, out_channels, kernel_size, stride, padding):
        zqs_train = self.VQVAE.get_conv2d_zqs(self.train_data_loader, in_channels, out_channels, kernel_size, stride, padding).numpy()
        zqs_test = self.VQVAE.get_conv2d_zqs(self.test_data_loader, in_channels, out_channels, kernel_size, stride, padding).numpy()
        if self.concatenate: return np.concatenate((zqs_train, zqs_test), axis=0)
        return zqs_train, zqs_test
    
    def flatten_zqs_indicies(self):
        _, s_train = self.VQVAE.get_flatten_zqs_s(self.train_data_loader)
        _, s_test = self.VQVAE.get_flatten_zqs_s(self.test_data_loader)
        if self.concatenate: return np.concatenate((s_train, s_test), axis=0)
        return s_train, s_test
    