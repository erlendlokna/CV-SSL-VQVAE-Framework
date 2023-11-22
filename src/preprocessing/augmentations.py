import numpy as np
from scipy import interpolate
import librosa
import scipy
from scipy.signal import find_peaks
from src.utils import time_to_timefreq, timefreq_to_time
import torch
from torch.distributions import uniform

import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from scipy.interpolate import interp1d

class Augmentations(object):
    def __init__(self, AmpR_rate=0.1, slope_rate=0.001, n_fft=2048, phase_max_change=np.pi/4, jitter_std=0.01, **kwargs):
        """
        :param AmpR_rate: rate for the `random amplitude resize`.
        """
        self.AmpR_rate = AmpR_rate
        self.slope_rate = slope_rate
        self.n_fft = n_fft  # Size of the FFT window
        self.phase_max_change = phase_max_change  # Maximum phase change
        self.jitter_std = jitter_std

    def amplitude_resize(self, *subx_views):
        """
        :param subx_view: (n_channels * subseq_len)
        """
        new_subx_views = []
        n_channels = subx_views[0].shape[0]
        for i in range(len(subx_views)):
            mul_AmpR = 1 + np.random.normal(0, self.AmpR_rate, size=(n_channels, 1))
            new_subx_view = subx_views[i] * mul_AmpR
            new_subx_views.append(new_subx_view)

        if len(new_subx_views) == 1:
            new_subx_views = new_subx_views[0]
        return new_subx_views
    
    def flip(self, *subx_views):
        """
        Randomly flip the input sequences horizontally.
        """
        flipped_subx_views = [np.flip(subx, axis=-1) if np.random.choice([True, False]) else subx for subx in subx_views]
        if len(flipped_subx_views) == 1:
            flipped_subx_views = flipped_subx_views[0]
        return flipped_subx_views
    
    def add_slope(self, *subx_views):
        """
        Add a linear slope to the input sequences.
        """
        sloped_subx_views = []
        for subx in subx_views:
            n_channels, subseq_len = subx.shape
            slope = np.random.uniform(-self.slope_rate, self.slope_rate, size=(n_channels, 1))
            x = np.arange(subseq_len)
            slope_component = slope * x
            sloped_subx = subx + slope_component
            sloped_subx_views.append(sloped_subx)

        if len(sloped_subx_views) == 1:
            sloped_subx_views = sloped_subx_views[0]
        return sloped_subx_views
    
    def stft_augmentation(self, *subx_views):
        """
        Apply STFT augmentation to the input sequences using PyTorch's STFT.
        """
        augmented_subx_views = []
        for subx in subx_views:
            n_channels, subseq_len = subx.shape
            augmented_subx = torch.zeros((n_channels, subseq_len))

            for i in range(n_channels):
                subx_tensor = torch.tensor(subx[i], dtype=torch.float32)

                n_fft = 2 ** int(np.ceil(np.log2(len(subx_tensor))))

                # Compute the STFT of the original data
                stft = torch.stft(subx_tensor, n_fft=n_fft, return_complex=True, onesided=False)

                # Modify the phase of the STFT representation while controlling phase changes
                magnitude = torch.abs(stft)
                phase = torch.angle(stft)
                phase_change = torch.empty_like(phase).uniform_(-self.phase_max_change, self.phase_max_change)
                augmented_phase = phase + phase_change

                # Reconstruct the augmented STFT
                augmented_stft = magnitude * torch.exp(1j * augmented_phase)

                # Inverse STFT to get the augmented signal
                augmented_signal = torch.istft(augmented_stft, n_fft = n_fft, return_complex=False, length=subseq_len, onesided=False)

                augmented_subx[i] = augmented_signal

            augmented_subx_views.append(augmented_subx)

        if len(augmented_subx_views) == 1:
            augmented_subx_views = augmented_subx_views[0]

        return np.array(augmented_subx_views)
    
    def jitter(self, *subx_views):
        """
        Add random jitter (noise) to each data point in the input sequence.
        """
        jittered_subx_views = []
        for subx in subx_views:
            jitter = np.random.normal(0, self.jitter_std, subx.shape)
            jittered_subx = subx + jitter
            jittered_subx_views.append(jittered_subx)

        if len(jittered_subx_views) == 1:
            jittered_subx_views = jittered_subx_views[0]
        return jittered_subx_views
    
    def time_slicing(self, *subx_views, slice_rate=0.2, p=0.3, expected_length=None):
        """
        Perform window slicing on the input sequences and pad to match expected length.
        :param subx_views: tuple of arrays, each of shape (n_channels, subseq_len)
        :param slice_rate: fraction of the sequence length to be used as the window size for slicing
        :param p: probability of applying time slicing
        :param expected_length: the expected sequence length for padding
        """

        sliced_subx_views = []
        for subx in subx_views:
            # Apply slicing only with probability p
            if np.random.rand() < p:
                n_channels, subseq_len = subx.shape
                window_size = int(subseq_len * slice_rate)
                
                # Randomly choose a start point for the slice
                start_point = np.random.randint(0, subseq_len - window_size + 1)
                end_point = start_point + window_size
                
                # Slice the sequence
                sliced_subx = subx[:, start_point:end_point]

                # If an expected length is provided, pad the sequence
                if expected_length is not None and window_size < expected_length:
                    padding = expected_length - window_size
                    # Here we use zero padding, but other strategies could be implemented
                    sliced_subx = np.pad(sliced_subx, ((0, 0), (0, padding)), 'constant', constant_values=0)
                
                sliced_subx_views.append(sliced_subx)
            else:
                # If not applying augmentation, just append the original sequence
                sliced_subx_views.append(subx)
        
        # If there was only one sequence, don't return a list
        if len(sliced_subx_views) == 1:
            return sliced_subx_views[0]
        
        return sliced_subx_views