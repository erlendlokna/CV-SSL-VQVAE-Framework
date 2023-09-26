import numpy as np
from einops import rearrange
import torch
import yaml


def compute_downsample_rate(input_length: int,
                            n_fft: int,
                            downsampled_width: int):
    return round(input_length / (np.log2(n_fft) - 1) / downsampled_width) if input_length >= downsampled_width else 1

def load_yaml_param_settings(yaml_fname: str):
    """
    :param yaml_fname: .yaml file that consists of hyper-parameter settings.
    """
    stream = open(yaml_fname, 'r')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    return config


def time_to_timefreq(x, n_fft: int, C: int):
    """
    x: (B, C, L)
    """
    x = rearrange(x, 'b c l -> (b c) l')
    x = torch.stft(x, n_fft, normalized=False, return_complex=True)
    x = torch.view_as_real(x)  # (B, N, T, 2); 2: (real, imag)
    x = rearrange(x, '(b c) n t z -> b (c z) n t ', c=C)  # z=2 (real, imag)
    return x  # (B, C, H, W)


def timefreq_to_time(x, n_fft: int, C: int):
    x = rearrange(x, 'b (c z) n t -> (b c) n t z', c=C).contiguous()
    x = torch.view_as_complex(x)
    x = torch.istft(x, n_fft, normalized=False, return_complex=False)
    x = rearrange(x, '(b c) l -> b c l', c=C)
    return x


def quantize(z, vq_model, transpose_channel_length_axes=False):
    input_dim = len(z.shape) - 2
    if input_dim == 2:
        h, w = z.shape[2:]
        z = rearrange(z, 'b c h w -> b (h w) c')
        z_q, indices, vq_loss, perplexity = vq_model(z)
        z_q = rearrange(z_q, 'b (h w) c -> b c h w', h=h, w=w)
    elif input_dim == 1:
        if transpose_channel_length_axes:
            z = rearrange(z, 'b c l -> b (l) c')
        z_q, indices, vq_loss, perplexity = vq_model(z)
        if transpose_channel_length_axes:
            z_q = rearrange(z_q, 'b (l) c -> b c l')
    else:
        raise ValueError
    return z_q, indices, vq_loss, perplexity

def freeze(model):
    for param in model.parameters():
        param.requires_grad = False
