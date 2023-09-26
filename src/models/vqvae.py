from src.models.encoder_decoder import VQVAEEncoder, VQVAEDecoder
from src.models.vq import VectorQuantize

from src.utils import (compute_downsample_rate,
                        time_to_timefreq,
                        timefreq_to_time,
                        quantize,
                        freeze)

import numpy as np

from src.models.base_model import BaseModel
from supervised_FCN.example_pretrained_model_loading import load_pretrained_FCN

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

import wandb

class VQVAE(BaseModel):
    def __init__(self,
                 input_length,
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

        if config['VQ-VAE']['perceptual_loss_weight']:
            self.fcn = load_pretrained_FCN(config['dataset']['dataset_name']).to(self.device)
            self.fcn.eval()
            freeze(self.fcn)

    def forward(self, batch):
        x, y = batch

        recons_loss = {'time': 0., 'timefreq': 0., 'perceptual': 0.}
        vq_loss = None
        perplexity = 0.

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

        if self.config['VQ-VAE']['perceptual_loss_weight']:
            z_fcn = self.fcn(x.float(), return_feature_vector=True).detach()
            zhat_fcn = self.fcn(xhat.float(), return_feature_vector=True)
            recons_loss['perceptual'] = F.mse_loss(z_fcn, zhat_fcn)

        # plot `x` and `xhat`
        r = np.random.rand()
        if self.training and r <= 0.05:
            b = np.random.randint(0, x.shape[0])
            c = np.random.randint(0, x_h.shape[1])

            fig, axes = plt.subplots(3, 1, figsize=(4, 2*3))
            plt.suptitle(f'ep_{self.current_epoch}')
            axes[0].plot(x_l[b, c].cpu())
            axes[0].plot(xhat_l[b, c].detach().cpu())
            axes[0].set_title('x_l')
            axes[0].set_ylim(-4, 4)

            axes[1].plot(x_h[b, c].cpu())
            axes[1].plot(xhat_h[b, c].detach().cpu())
            axes[1].set_title('x_h')
            axes[1].set_ylim(-4, 4)

            axes[2].plot(x_l[b, c].cpu() + x_h[b, c].cpu())
            axes[2].plot(xhat_l[b, c].detach().cpu() + xhat_h[b, c].detach().cpu())
            axes[2].set_title('x')
            axes[2].set_ylim(-4, 4)

            plt.tight_layout()
            wandb.log({"x vs xhat (training)": wandb.Image(plt)})
            plt.close()

        return recons_loss, vq_losses, perplexities
