import numpy as np
from scipy import interpolate
import librosa
import scipy
from scipy.signal import find_peaks


class Augmentations(object):
    def __init__(self, AmpR_rate=0.1, slope_rate=0.001, n_fft=2048, hop_length=512, phase_max_change=np.pi/4, **kwargs):
        """
        :param AmpR_rate: rate for the `random amplitude resize`.
        """
        self.AmpR_rate = AmpR_rate
        self.slope_rate = slope_rate
        self.n_fft = n_fft  # Size of the FFT window
        self.hop_length = hop_length  # Hop length for STFT
        self.phase_max_change = phase_max_change  # Maximum phase change
        self.iaaft_iterations = 50

    def random_crop(self, subseq_len: int, *x_views):
        subx_views = []
        rand_ts = []
        for i in range(len(x_views)):
            seq_len = x_views[i].shape[-1]
            rand_t = np.random.randint(0, seq_len - subseq_len + 1, size=1)[0]
            subx = x_views[i][:, rand_t: rand_t + subseq_len]  # (subseq_len)
            subx_views.append(subx)
            rand_ts.append(rand_t)

        if len(subx_views) == 1:
            subx_views = subx_views[0]
        return subx_views

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
        Apply STFT augmentation to the input sequences.
        """
        augmented_subx_views = []
        for subx in subx_views:
            n_channels, subseq_len = subx.shape
            augmented_subx = np.zeros((n_channels, subseq_len))

            for i in range(n_channels):

                n_fft = 2**int(np.ceil(np.log2(len(subx[i]))))
                    
                # Compute the STFT of the original data
                stft = librosa.stft(subx[i], n_fft=n_fft, hop_length=self.hop_length)

                # Modify the phase of the STFT representation while controlling phase changes
                magnitude = np.abs(stft)
                phase = np.angle(stft)
                phase_change = np.random.uniform(-self.phase_max_change, self.phase_max_change, size=phase.shape)
                augmented_phase = phase + phase_change

                # Reconstruct the augmented STFT
                augmented_stft = magnitude * np.exp(1j * augmented_phase)

                # Inverse STFT to get the augmented signal
                augmented_signal = librosa.istft(augmented_stft, hop_length=self.hop_length)

                augmented_subx[i] = augmented_signal[:subseq_len]

            augmented_subx_views.append(augmented_subx)

        if len(augmented_subx_views) == 1:
            augmented_subx_views = augmented_subx_views[0]

        return augmented_subx_views
    
    
