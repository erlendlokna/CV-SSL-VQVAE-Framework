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
from src.experiments.tests import svm_test, knn_test, intristic_dimension, multiple_tests
from sklearn.decomposition import PCA

class VQVAE(BaseModel):
    def __init__(self,
                 input_length,
                 test_data_loader,
                 train_data_loader,
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

        if config['VQVAE']['perceptual_loss_weight']:
            self.fcn = load_pretrained_FCN(config['dataset']['dataset_name']).to(self.device)
            self.fcn.eval()
            freeze(self.fcn)
        
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        
    def forward(self, batch):      
        x, y = batch

        recons_loss = {'time': 0., 'timefreq': 0., 'perceptual': 0.}
        vq_loss = None
        perplexity = 0.

        #forward
        C = x.shape[1]
        u = time_to_timefreq(x, self.n_fft, C)  # (B, C, H, W)
        
        if not self.decoder.is_upsample_size_updated:
                self.decoder.register_upsample_size(torch.IntTensor(np.array(u.shape[2:])))


        z = self.encoder(u)

        z_q, indices, vq_loss, perplexity = quantize(z, self.vq_model)


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
        if self.training and r <= 0.008:
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


        return recons_loss, vq_loss, perplexity
    

    def training_step(self, batch, batch_idx):
        x = batch
        recons_loss, vq_loss, perplexity = self.forward(x)

        """
        print('-----------------')
        print("recons_loss['time']:", type(recons_loss['time']))
        print("recons_loss['timefreq']", type(recons_loss['timefreq']))
        print("vq_loss", vq_loss)
        print("recons_loss['perceptual']:", type(recons_loss['perceptual']))
        print('-----------------')
        """

        loss = recons_loss['time'] + recons_loss['timefreq'] + vq_loss['loss'] + recons_loss['perceptual']
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

                     'perceptual': recons_loss['perceptual']
                     }
        
        wandb.log(loss_hist)

        detach_the_unnecessary(loss_hist)
        return loss_hist
    
    def validation_step(self, batch, batch_idx):
        x = batch
        recons_loss, vq_loss, perplexity = self.forward(x)

        loss = recons_loss['time'] + recons_loss['timefreq'] + vq_loss['loss'] + recons_loss['perceptual']

        # log
        val_loss_hist = {'validation_loss': loss,
                     'validation_recons_loss.time': recons_loss['time'],

                     'validation_recons_loss.timefreq': recons_loss['timefreq'],

                     'validation_commit_loss': vq_loss['commit_loss'],
                     #'validation_commit_loss': vq_loss, #?
                     
                     'validation_perplexity': perplexity,

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
        recons_loss, vq_loss, perplexity = self.forward(x)

        loss = recons_loss['time'] + recons_loss['timefreq'] + vq_loss['loss'] + recons_loss['perceptual']
        
        # log
        loss_hist = {'loss': loss,
                     'recons_loss.time': recons_loss['time'],

                     'recons_loss.timefreq': recons_loss['timefreq'],

                     'commit_loss': vq_loss['commit_loss'],
                     #'commit_loss': vq_loss, #?
                     
                     'perplexity': perplexity,

                     'perceptual': recons_loss['perceptual']
                     }
        
        detach_the_unnecessary(loss_hist)
        return loss_hist
    
    # ---- Representation testing ------ 
    def on_train_epoch_end(self):
        if self.current_epoch % 100 == 0 and self.current_epoch != 0:
            self.test_representations()
        
        if self.current_epoch < 100 and self.current_epoch % 30 == 0:
            self.test_representations()

    def on_train_epoch_start(self):
        if self.current_epoch == 0:
            self.test_representations()

    def test_representations(self):
        zqs_train, _ = self.run_through_encoder_codebook(self.train_data_loader) #non augmented versions
        zqs_train = torch.flatten(zqs_train.detach(), start_dim=1)
        y_train = self.train_data_loader.dataset.Y.flatten().astype(int)
        
        zqs_test, _ = self.run_through_encoder_codebook(self.test_data_loader)
        zqs_test = torch.flatten(zqs_test.detach(), start_dim=1)
        y_test = self.test_data_loader.dataset.Y.flatten().astype(int)

        Z = np.concatenate((zqs_train, zqs_test), axis=0)
        Y = np.concatenate((y_train, y_test), axis=0)
        print("intrinstic dimension...")
        intristic_dim = intristic_dimension(Z)
        
        print("probe tests...")
        svm_acc = np.mean(multiple_tests(test=svm_test, Z=(zqs_train, zqs_test), Y=(y_train, y_test), n_runs=1))
        #classnet_acc = np.mean(multiple_tests(classnet_test, (zqs_train, zqs_test), (y_train, y_test), n_runs=4))
        knn_acc = np.mean(multiple_tests(test=knn_test, Z=(zqs_train, zqs_test), Y=(y_train, y_test), n_runs=1))

        reps = {
            'intrinstic_dim': intristic_dim,
            'svm_acc': svm_acc,
            'knn_acc': knn_acc
        }
        wandb.log(reps)
        print("PCA...")
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
            

