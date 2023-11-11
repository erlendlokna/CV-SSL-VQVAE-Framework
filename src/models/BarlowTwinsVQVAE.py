import numpy as np
import matplotlib.pyplot as plt

from src.models.encoder_decoder import VQVAEEncoder, VQVAEDecoder
from src.models.vq import VectorQuantize

from src.utils import (compute_downsample_rate,

                        time_to_timefreq,
                        timefreq_to_time,
                        quantize,
                        )

from src.models.base_model import BaseModel, detach_the_unnecessary
from supervised_FCN.example_pretrained_model_loading import load_pretrained_FCN

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import umap
import wandb
from src.experiments.tests import svm_test, knn_test, svm_test_gs_rbf, intristic_dimension, standard_scale, minmax_scale
from sklearn.decomposition import PCA
from src.models.barlowtwins import BarlowTwins, Projector


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

        #projector
        projector = Projector(last_channels_enc=dim, proj_hid=config['barlow_twins']['proj_hid'], proj_out=config['barlow_twins']['proj_out'], 
                              device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        #barlow twins loss function
        self.barlow_twins = BarlowTwins(projector, lambda_=0.005)

        #save these for representation learning tests during training
        self.train_data_loader = non_aug_train_data_loader
        self.test_data_loader = non_aug_test_data_loader


    def forward(self, batch):      
        subxs_pairs, y = batch
        if len(subxs_pairs[0]) == 2:
            x, x_aug = subxs_pairs[0] #load two augmented x's
            #x1 is the non-augmented timeseries
        else:
            x = x_aug = subxs_pairs #in case for validation
            
        recons_loss = {'time': 0., 'timefreq': 0., 'perceptual': 0.}
        vq_losses = None
        perplexity = 0.

        #forward
        C = x.shape[1]
        u = time_to_timefreq(x, self.n_fft, C)  # (B, C, H, W)
        u_aug = time_to_timefreq(x_aug, self.n_fft, C)
        
        if not self.decoder.is_upsample_size_updated:
            self.decoder.register_upsample_size(torch.IntTensor(np.array(u.shape[2:])))

        z = self.encoder(u)
        z_aug = self.encoder(u_aug)

        barlow_twins_loss = self.barlow_twins(z, z_aug)

        z_q, indices, vq_loss, perp = quantize(z, self.vq_model)
        z_q_aug, indices_aug, vq_loss_aug, perp_aug = quantize(z_aug, self.vq_model)

        barlow_twins_loss = self.barlow_twins(z_q, z_q_aug)

        picked_aug = False
        if np.random.uniform(0, 1) < 0.5:
            uhat = self.decoder(z_q_aug)
            xhat = timefreq_to_time(uhat, self.n_fft, C)
            recons_loss['time'] = F.mse_loss(x_aug, xhat)
            recons_loss['timefreq'] = F.mse_loss(u_aug, uhat)
            picked_aug = True
        else: 
            uhat = self.decoder(z_q)
            xhat = timefreq_to_time(uhat, self.n_fft, C)
            recons_loss['time'] = F.mse_loss(x, xhat)
            recons_loss['timefreq'] = F.mse_loss(u, uhat)

        vq_losses = {}
        for key in vq_loss.keys():
            vq_losses[key] = 0.5 * (vq_loss[key] + vq_loss_aug[key])
        perplexity = 0.5 * (perp + perp_aug)

        # plot `x` and `xhat`
        r = np.random.uniform(0, 1)
        if r < 0.01:
            b = np.random.randint(0, x.shape[0])
            c = np.random.randint(0, x.shape[1])
            fig, ax = plt.subplots()
            plt.suptitle(f'ep_{self.current_epoch}')
            
            ax.plot(x[b, c].cpu())
            label = "x_aug_hat" if picked_aug else "xhat"
            ax.plot(xhat[b,c].detach().cpu(), label=f"{label}")
            ax.plot(x_aug[b, c].detach().cpu(), '--', c='grey')
            ax.set_title('x')
            ax.set_ylim(-4, 4)
            fig.legend()
            wandb.log({"augmented x vs xhat (with augmented x)": wandb.Image(plt)})
            plt.close()

        return recons_loss, vq_losses, perplexity, barlow_twins_loss
    

    def training_step(self, batch, batch_idx):
        x = batch

        recons_loss, vq_loss, perplexity, barlow_twins_loss = self.forward(x)

        loss = recons_loss['time'] + recons_loss['timefreq'] + vq_loss['loss'] + recons_loss['perceptual'] +  1.2 * barlow_twins_loss
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

                     'barrow_twins_loss': barlow_twins_loss
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
        if self.current_epoch % 100 == 0 and self.current_epoch != 0 :
            self.test_representations()
        if self.current_epoch == 2000:
            self.test_representations()
        
        if self.current_epoch < 100 and self.current_epoch % 20 == 0:
            self.test_representations()

    def on_train_epoch_start(self):
        if self.current_epoch == 0:
            self.test_representations()

    def test_representations(self):
        print("Grabbing latent variables")
        ztr, ytr = self.encode_data(self.train_data_loader, self.encoder)
        zts, yts = self.encode_data(self.test_data_loader, self.encoder)
        print("projecting..")
        ztr = self.barlow_twins.projector(ztr)
        zts = self.barlow_twins.projector(zts)

        ztr = torch.flatten(ztr, start_dim=1).detach().cpu().numpy()
        zts = torch.flatten(zts, start_dim=1).detach().cpu().numpy()
        ytr = torch.flatten(ytr, start_dim=0).detach().cpu().numpy()
        yts = torch.flatten(yts, start_dim=0).detach().cpu().numpy()

        ztr, zts = minmax_scale(ztr, zts)

        z = np.concatenate((ztr, zts), axis=0)
        y = np.concatenate((ytr, yts), axis=0)
        
        intristic_dim = intristic_dimension(z.reshape(-1, z.shape[-1]))
        svm_acc = svm_test(ztr, zts, ytr, yts)
        svm_gs_rbf_acc = 0.0#svm_test_gs_rbf(ztr, zts, ytr, yts)
        knn1_acc, knn5_acc, knn10_acc = knn_test(ztr, zts, ytr, yts)

        wandb.log({
            'intrinstic_dim': intristic_dim,
            'svm_acc': svm_acc,
            'svm_rbf': svm_gs_rbf_acc,
            'knn1_acc': knn1_acc,
            'knn5_acc': knn5_acc,
            'knn10_acc': knn10_acc
        })

        embs = PCA(n_components=2).fit_transform(z)
        f, a = plt.subplots()
        plt.suptitle(f'ep_{self.current_epoch}')
        a.scatter(embs[:, 0], embs[:, 1], c=y)
        wandb.log({"PCA plot": wandb.Image(f)})
        plt.close()

    
    def encode_data(self, dataloader, encoder, cuda=True):
        z_list = []  # List to hold all the encoded representations
        y_list = []  # List to hold all the labels/targets

        # Iterate over the entire dataloader
        for batch in dataloader:
            x, y = batch  # Unpack the batch.

            # Perform the encoding
            if cuda:
                x = x.cuda()
            C = x.shape[1]
            xf = time_to_timefreq(x, self.n_fft, C).to(x.device)  # Convert time domain to frequency domain
            z = encoder(xf)  # Encode the input

            # Convert the tensors to lists and append to z_list and y_list
            z_list.extend(z.cpu().detach().tolist())
            y_list.extend(y.cpu().detach().tolist())  # Make sure to detach y and move to CPU as well

        # Convert lists of lists to 2D tensors
        z_encoded = torch.tensor(z_list)
        ys = torch.tensor(y_list)
        if cuda:
            z_encoded = z_encoded.cuda()
            ys = ys.cuda()

        # Flatten the tensor to 2D if it's not already
        #z_encoded = z_encoded.view(z_encoded.size(0), -1)

        return z_encoded, ys
    
