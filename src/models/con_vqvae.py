import numpy as np
import matplotlib.pyplot as plt

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

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_metric_learning.losses import ContrastiveLoss
#from src.models.contrastiveloss import NTXentLoss
from pytorch_metric_learning.losses import NTXentLoss
from pathlib import Path
import tempfile

import wandb
from src.experiments.tests import svm_test, knn_test, classnet_test, intristic_dimension, multiple_tests
from sklearn.decomposition import PCA
from src.models.barlowtwinsloss import BarlowTwinsLoss

class InfoNCE_VQVAE(BaseModel):
    def __init__(self,
                 input_length,
                 test_data_loader,
                 train_data_loader,
                 config: dict,
                 n_train_samples: int):
        super().__init__()

        self.config = config
        self.T_max = config['trainer_params']['max_epochs']['vqvae'] * (np.ceil(n_train_samples / config['dataset']['batch_sizes']['vqvae']) + 1)
        
        self.n_fft = config['VQVAE']['n_fft']
        dim = config['encoder']['dim']
        in_channels = config['dataset']['in_channels']

        downsampled_width = config['encoder']['downsampled_width']
        downsampled_rate = compute_downsample_rate(input_length, self.n_fft, downsampled_width)

        #encoder
        self.encoder = VQVAEEncoder(dim, 2*in_channels, downsampled_rate, config['encoder']['n_resnet_blocks'])
        
        #vector quantiser
        self.vq_model = VectorQuantize(dim, config['VQVAE']['codebook']['size'], **config['VQVAE'])

        #decoder
        self.decoder = VQVAEDecoder(dim, 2 * in_channels, downsampled_rate, config['decoder']['n_resnet_blocks'])

        if config['VQVAE']['perceptual_loss_weight']:
            self.fcn = load_pretrained_FCN(config['dataset']['dataset_name']).to(self.device)
            self.fcn.eval()
            freeze(self.fcn)

        self.conloss = NTXentLoss(temperature=0.07)

        self.epoch_count = 0 # for representation learning
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader

    
    def forward(self, batch):      
        x, y = batch

        recons_loss = {'time': 0., 'timefreq': 0., 'perceptual': 0.}
        vq_loss = None
        perplexity = 0.
        contrastive_loss = 0.

        #forward
        C = x.shape[1]
        u = time_to_timefreq(x, self.n_fft, C)  # (B, C, H, W)
        
        if not self.decoder.is_upsample_size_updated:
                self.decoder.register_upsample_size(torch.IntTensor(np.array(u.shape[2:])))


        z = self.encoder(u)

        # ---> Contrastive loss <----
        """
        contrastive_loss = self.contrastive_loss_func(
            z.view(z.shape[0], -1), 
            torch.flatten(y, start_dim=0)
        )
        """
        #contrastive_loss = self.contrastive_criterion(
        #    z.view(z.shape[0], -1), torch.flatten(y, start_dim=0)
        #)

        z_q, indices, vq_loss, perplexity = quantize(z, self.vq_model)
        
        """
        # Get the dimensions of 'z_q'
        batch_size, sequence_length, embedding_dim, codebook_size = z_q.size()

        # Reshape 'z_q' for easy manipulation
        z_q_c = z_q.reshape(batch_size, -1, codebook_size)  # Reshaped to (batch_size, sequence_length * embedding_dim, codebook_size)
        z_q_c = F.normalize(z_q_c, dim=2, p=2)
        # Shuffle the 'z_q' tensor along the batch dimension (create 'z_q2')
        z_q2 = z_q_c[torch.randperm(batch_size)]

        # Create 'z_q1' by duplicating 'z_q'
        z_q1 = z_q_c

        contrastive_loss = self.conloss(z_q1, z_q2)
        """

        batch_size, sequence_length, embedding_dim, codebook_size = z_q.size()

        # Reshape 'z_q' for easy manipulation
        z_qc = torch.flatten(z_q, start_dim=1)

        contrastive_loss = self.conloss(z_qc, torch.flatten(y, start_dim=0))

        uhat = self.decoder(z_q)
        xhat = timefreq_to_time(uhat, self.n_fft, C)

        recons_loss['time'] = F.mse_loss(x, xhat)
        recons_loss['timefreq'] = F.mse_loss(u, uhat)
        #perplexity = perplexity #Updated above during quantize
        #vq_losses['LF'] = vq_loss_l #Updated above during quantize

        if self.config['VQVAE']['perceptual_loss_weight']:
            z_fcn = self.fcn(x.float(), return_feature_vector=True).detach()
            zhat_fcn = self.fcn(xhat.float(), return_feature_vector=True)
            recons_loss['perceptual'] = F.mse_loss(z_fcn, zhat_fcn)

        # plot `x` and `xhat`
        r = np.random.rand()
        if self.training and r <= 0.05:
            b = np.random.randint(0, x.shape[0])
            c = np.random.randint(0, x.shape[1])
            fig, ax = plt.subplots()
            plt.suptitle(f'ep_{self.current_epoch}')
            ax.plot(x[b, c].cpu())
            ax.plot(xhat[b,c].detach().cpu())
            ax.set_title('x')
            ax.set_ylim(-4, 4)

            wandb.log({"x vs xhat (training)": wandb.Image(plt)})
            plt.close()


        return recons_loss, vq_loss, perplexity, contrastive_loss
    

    def training_step(self, batch, batch_idx):
        x = batch
        recons_loss, vq_loss, perplexity, contrastive_loss = self.forward(x)

        """
        print('-----------------')
        print("recons_loss['time']:", type(recons_loss['time']))
        print("recons_loss['timefreq']", type(recons_loss['timefreq']))
        print("vq_loss", vq_loss)
        print("recons_loss['perceptual']:", type(recons_loss['perceptual']))
        print('-----------------')
        """

        loss = recons_loss['time'] + recons_loss['timefreq'] + vq_loss['loss'] + recons_loss['perceptual'] + contrastive_loss
        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        # log
        loss_hist = {'loss': loss,
                     'recons_loss.time': recons_loss['time'],

                     'recons_loss.timefreq': recons_loss['timefreq'],

                     'commit_loss': vq_loss['commit_loss'],
                     #'commit_loss': vq_loss, #?
                     
                     'perplexity': perplexity,

                     'contrastive': contrastive_loss,

                     'perceptual': recons_loss['perceptual']
                     }
        
        wandb.log(loss_hist)

        detach_the_unnecessary(loss_hist)
        return loss_hist
    
    def validation_step(self, batch, batch_idx):
        x = batch
        recons_loss, vq_loss, perplexity, contrastive_loss = self.forward(x)

        loss = recons_loss['time'] + recons_loss['timefreq'] + vq_loss['loss'] + recons_loss['perceptual'] + contrastive_loss

        # log
        val_loss_hist = {'validation_loss': loss,
                     'validation_recons_loss.time': recons_loss['time'],

                     'validation_recons_loss.timefreq': recons_loss['timefreq'],

                     'validation_commit_loss': vq_loss['commit_loss'],
                     #'validation_commit_loss': vq_loss, #?
                     
                     'validation_perplexity': perplexity,

                     'contrastive_loss': contrastive_loss,

                     'validation_perceptual': recons_loss['perceptual']
                     }
        
        detach_the_unnecessary(val_loss_hist)
        wandb.log(val_loss_hist)
        return val_loss_hist


    def configure_optimizers(self):
        opt = torch.optim.AdamW([{'params': self.encoder.parameters(), 'lr': self.config['model_params']['LR']},
                                 {'params': self.decoder.parameters(), 'lr': self.config['model_params']['LR']},
                                 {'params': self.vq_model.parameters(), 'lr': self.config['model_params']['LR']},
                                 ],
                                weight_decay=self.config['model_params']['weight_decay'])
        
        return {'optimizer': opt, 'lr_scheduler': CosineAnnealingLR(opt, self.T_max)}


    def test_step(self, batch, batch_idx):
        x = batch
        recons_loss, vq_loss, perplexity, contrastive_loss = self.forward(x)

        loss = recons_loss['time'] + recons_loss['timefreq'] + vq_loss['loss'] + recons_loss['perceptual']
        
        # log
        loss_hist = {'loss': loss,
                     'recons_loss.time': recons_loss['time'],

                     'recons_loss.timefreq': recons_loss['timefreq'],

                     'commit_loss': vq_loss['commit_loss'],
                     #'commit_loss': vq_loss, #?
                     
                     'perplexity': perplexity,

                     'contrastive': contrastive_loss,

                     'perceptual': recons_loss['perceptual']
                     }
        
        detach_the_unnecessary(loss_hist)
        return loss_hist
    
    # ---- Representation testing ------
    def on_train_epoch_end(self):
        if self.current_epoch %  100 == 0 and self.current_epoch != 0:
            self.test_representations()

    def on_train_epoch_start(self):
        if self.current_epoch == 0:
            self.test_representations()

    def test_representations(self):
        zqs_train, _ = self.run_through_encoder_codebook(self.train_data_loader)
        zqs_train = torch.flatten(zqs_train.detach(), start_dim=1)
        y_train = self.train_data_loader.dataset.Y.flatten().astype(int)
        zqs_test, _ = self.run_through_encoder_codebook(self.test_data_loader)
        zqs_test = torch.flatten(zqs_test.detach(), start_dim=1)
        y_test = self.test_data_loader.dataset.Y.flatten().astype(int)

        Z = np.concatenate((zqs_train, zqs_test), axis=0)
        Y = np.concatenate((y_train, y_test), axis=0)

        intristic_dim = intristic_dimension(Z)
        svm_acc = np.mean(multiple_tests(svm_test, Z, Y, n_runs=40))
        classnet_acc = np.mean(multiple_tests(classnet_test, Z, Y, n_runs=4))
        
        reps = {
            'intrinstic_dim': intristic_dim,
            'svm_acc': svm_acc,
            'classnet_acc': classnet_acc
        }
        wandb.log(reps)

        embs = PCA(n_components=2).fit_transform(Z)
        f, a = plt.subplots()
        plt.suptitle(f'ep_{self.current_epoch}')
        a.scatter(embs[:, 0], embs[:, 1], c=Y)
        wandb.log({"PCA plot": wandb.Image(f)})

    def run_through_encoder_codebook(self, data_loader):
        #collecting all the timeseries codebook index representations:
        dataloader_iterator = iter(data_loader)
        number_of_batches = len(data_loader)

        zqs_list = [] #TODO: make static. List containing zqs for each timeseries in data_loader
        s_list = []

        for i in range(number_of_batches):
            try:
                x, y = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(data_loader)
                x, y = next(dataloader_iterator)
            
            z_q, s = self.encode_to_z_q(x)

            for i, zq_i in enumerate(z_q):    
                zqs_list.append(zq_i.detach().tolist())
                s_list.append(s[i].tolist())

        zqs_tensor = torch.tensor(zqs_list, dtype=torch.float64)
        s_tensor = torch.tensor(s_list, dtype=torch.int32)
        return zqs_tensor, s_tensor

    def encode_to_z_q(self, x):
        """
        x: (B, C, L)
        """
        x = x.cuda()
        C = x.shape[1]
        xf = time_to_timefreq(x, self.n_fft, C)  # (B, C, H, W)
        
        z = self.encoder(xf)  # (b c h w)
        z_q, indices, vq_loss, perplexity = quantize(z, self.vq_model)  # (b c h w), (b (h w) h), ...
        return z_q, indices
    

class BarlowTwinsVQVAE(BaseModel):
    #requires the augmented batches.
    def __init__(self,
                 input_length,
                 non_aug_test_data_loader,
                 non_aug_train_data_loader,
                 config: dict,
                 n_train_samples: int,
                ):
        super().__init__()

        self.config = config
        self.T_max = config['trainer_params']['max_epochs']['vqvae'] * (np.ceil(n_train_samples / config['dataset']['batch_sizes']['vqvae']) + 1)
        
        self.n_fft = config['VQVAE']['n_fft']
        dim = config['encoder']['dim']
        in_channels = config['dataset']['in_channels']

        downsampled_width = config['encoder']['downsampled_width']
        downsampled_rate = compute_downsample_rate(input_length, self.n_fft, downsampled_width)

        #encoder
        self.encoder = VQVAEEncoder(dim, 2*in_channels, downsampled_rate, config['encoder']['n_resnet_blocks'])
        
        #vector quantiser
        self.vq_model = VectorQuantize(dim, config['VQVAE']['codebook']['size'], **config['VQVAE'])

        #decoder
        self.decoder = VQVAEDecoder(dim, 2 * in_channels, downsampled_rate, config['decoder']['n_resnet_blocks'])

        self.barlowtwinsloss = BarlowTwinsLoss(0.005)
        self.na_train_data_loader = non_aug_train_data_loader
        self.na_test_data_loader = non_aug_test_data_loader
        
    def forward(self, batch):      
        subxs_pairs, y = batch
        random_pair_index = np.random.randint(0, len(subxs_pairs))
        if len(subxs_pairs) == 2:
            x1, x2 = subxs_pairs[random_pair_index] #load two augmented x's
        else:
            x1 = x2 = subxs_pairs #in case for validation
            
        recons_loss = {'time': 0., 'timefreq': 0., 'perceptual': 0.}
        vq_loss = None
        perplexity = 0.

        #forward
        C = x1.shape[1]
        u1 = time_to_timefreq(x1, self.n_fft, C)  # (B, C, H, W)
        u2 = time_to_timefreq(x2, self.n_fft, C)
        
        if not self.decoder.is_upsample_size_updated:
                self.decoder.register_upsample_size(torch.IntTensor(np.array(u1.shape[2:])))

        z1 = self.encoder(u1)
        z2 = self.encoder(u2)
        z_q1, indices1, vq_loss1, perplexity1 = quantize(z1, self.vq_model)
        z_q2, indices2, vq_loss2, perplexity2 = quantize(z2, self.vq_model)

        
        barrow_twins_loss = self.barlowtwinsloss(
            z_q1, z_q2
        )

        uhat1 = self.decoder(z_q1)
        uhat2 = self.decoder(z_q2)

        xhat1 = timefreq_to_time(uhat1, self.n_fft, C)
        xhat2 = timefreq_to_time(uhat2, self.n_fft, C)

        recons_loss['time'] = 0.5*(F.mse_loss(x1, xhat1) + F.mse_loss(x2, xhat2))
        recons_loss['timefreq'] = 0.5 * (F.mse_loss(u1, uhat1) + F.mse_loss(u2, uhat2))
        vq_loss = {}
        for key in vq_loss1.keys(): vq_loss[key] = 0.5 * (vq_loss1[key] + vq_loss2[key])
        perplexity = 0.5 * (perplexity1 + perplexity2)

        # plot `x` and `xhat`
        r = np.random.rand()
        if self.training and r <= 0.05:
            b = np.random.randint(0, x1.shape[0])
            c = np.random.randint(0, x1.shape[1])
            fig, ax = plt.subplots()
            plt.suptitle(f'ep_{self.current_epoch}')
            ax.plot(x1[b, c].cpu())
            ax.plot(xhat1[b,c].detach().cpu())
            ax.set_title('x')
            ax.set_ylim(-4, 4)

            wandb.log({"x1 vs xhat1 (training)": wandb.Image(plt)})
            plt.close()

        return recons_loss, vq_loss, perplexity, barrow_twins_loss
    

    def training_step(self, batch, batch_idx):
        x = batch

        recons_loss, vq_loss, perplexity, barrow_twins_loss = self.forward(x)

        loss = recons_loss['time'] + recons_loss['timefreq'] + vq_loss['loss'] + recons_loss['perceptual'] + barrow_twins_loss
        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        # log
        loss_hist = {'loss': loss,
                     'recons_loss.time': recons_loss['time'],

                     'recons_loss.timefreq': recons_loss['timefreq'],

                     'commit_loss': vq_loss['commit_loss'],
                     #'commit_loss': vq_loss, #?
                     
                     'perplexity': perplexity,

                     'perceptual': recons_loss['perceptual'],

                     'barrow_twins_loss': barrow_twins_loss
                     }
        
        wandb.log(loss_hist)

        detach_the_unnecessary(loss_hist)
        return loss_hist
    
    def validation_step(self, batch, batch_idx):
        x = batch
        recons_loss, vq_loss, perplexity, barrow_twins_loss = self.forward(x)

        loss = recons_loss['time'] + recons_loss['timefreq'] + vq_loss['loss'] + recons_loss['perceptual']

        # log
        val_loss_hist = {'validation_loss': loss,
                     'validation_recons_loss.time': recons_loss['time'],

                     'validation_recons_loss.timefreq': recons_loss['timefreq'],

                     'validation_commit_loss': vq_loss['commit_loss'],
                     #'validation_commit_loss': vq_loss, #?
                     
                     'validation_perplexity': perplexity,

                     'validation_perceptual': recons_loss['perceptual'],

                     'barrow_twins_loss': barrow_twins_loss
                     }
        
        detach_the_unnecessary(val_loss_hist)
        wandb.log(val_loss_hist)

        return val_loss_hist


    def configure_optimizers(self):
        opt = torch.optim.AdamW([{'params': self.encoder.parameters(), 'lr': self.config['model_params']['LR']},
                                 {'params': self.decoder.parameters(), 'lr': self.config['model_params']['LR']},
                                 {'params': self.vq_model.parameters(), 'lr': self.config['model_params']['LR']},
                                 ],
                                weight_decay=self.config['model_params']['weight_decay'])
        
        return {'optimizer': opt, 'lr_scheduler': CosineAnnealingLR(opt, self.T_max)}


    def test_step(self, batch, batch_idx):
        x = batch
        recons_loss, vq_loss, perplexity, barrow_twins_loss = self.forward(x)

        loss = recons_loss['time'] + recons_loss['timefreq'] + vq_loss['loss'] + recons_loss['perceptual']
        
        # log
        loss_hist = {'loss': loss,
                     'recons_loss.time': recons_loss['time'],

                     'recons_loss.timefreq': recons_loss['timefreq'],

                     'commit_loss': vq_loss['commit_loss'],
                     #'commit_loss': vq_loss, #?
                     
                     'perplexity': perplexity,

                     'perceptual': recons_loss['perceptual'],

                     'barrow_twins_loss': barrow_twins_loss
                     }
        
        detach_the_unnecessary(loss_hist)
        return loss_hist
    
    # ---- Representation testing ------
    def on_train_epoch_end(self):
        if self.current_epoch == 0:
            self.test_representations()

    def on_train_epoch_start(self):
        if self.current_epoch % 100 == 0:
            self.test_representations()

    def test_representations(self):
        zqs_train, _ = self.run_through_encoder_codebook(self.na_train_data_loader) #non augmented versions
        zqs_train = torch.flatten(zqs_train.detach(), start_dim=1)
        y_train = self.na_train_data_loader.dataset.Y.flatten().astype(int)
        
        zqs_test, _ = self.run_through_encoder_codebook(self.na_test_data_loader)
        zqs_test = torch.flatten(zqs_test.detach(), start_dim=1)
        y_test = self.na_test_data_loader.dataset.Y.flatten().astype(int)

        Z = np.concatenate((zqs_train, zqs_test), axis=0)
        Y = np.concatenate((y_train, y_test), axis=0)

        intristic_dim = intristic_dimension(Z)
        svm_acc = np.mean(multiple_tests(test=svm_test, Z=(zqs_train, zqs_test), Y=(y_train, y_test), n_runs=1))
        #classnet_acc = np.mean(multiple_tests(classnet_test, (zqs_train, zqs_test), (y_train, y_test), n_runs=4))
        knn_acc = np.mean(multiple_tests(test=knn_test, Z=(zqs_train, zqs_test), Y=(y_train, y_test), n_runs=1))

        reps = {
            'intrinstic_dim': intristic_dim,
            'svm_acc': svm_acc,
            'knn_acc': knn_acc
        }
        wandb.log(reps)
        
        embs = PCA(n_components=2).fit_transform(Z)
        f, a = plt.subplots()
        plt.suptitle(f'ep_{self.current_epoch}')
        a.scatter(embs[:, 0], embs[:, 1], c=Y)
        wandb.log({"PCA plot": wandb.Image(f)})
        

    def run_through_encoder_codebook(self, data_loader):
        #collecting all the timeseries codebook index representations:
        dataloader_iterator = iter(data_loader)
        number_of_batches = len(data_loader)

        zqs_list = [] #TODO: make static. List containing zqs for each timeseries in data_loader
        s_list = []

        for i in range(number_of_batches):
            try:
                x, y = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(data_loader)
                x, y = next(dataloader_iterator)
            
            z_q, s = self.encode_to_z_q(x)
            
            for i, zq_i in enumerate(z_q):    
                zqs_list.append(zq_i.detach().tolist())
                s_list.append(s[i].tolist())

        zqs_tensor = torch.tensor(zqs_list, dtype=torch.float64)
        s_tensor = torch.tensor(s_list, dtype=torch.int32)
        return zqs_tensor, s_tensor

    def encode_to_z_q(self, x):
        """
        x: (B, C, L)
        """
        x = x.cuda()
        C = x.shape[1]
        xf = time_to_timefreq(x, self.n_fft, C)  # (B, C, H, W)
        
        z = self.encoder(xf)  # (b c h w)
        z_q, indices, vq_loss, perplexity = quantize(z, self.vq_model)  # (b c h w), (b (h w) h), ...
        return z_q, indices