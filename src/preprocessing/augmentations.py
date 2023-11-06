import numpy as np
from scipy import interpolate


class Augmentations(object):
    def __init__(self, AmpR_rate=0.1, jitter_std=0.05, **kwargs):
        """
        :param AmpR_rate: rate for the `random amplitude resize`.
        """
        self.AmpR_rate = AmpR_rate
        self.jitter_std = jitter_std


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

    def jitter(self, *x_views, noise_probability=0.1):
        """
        Apply jittering to the input time series.
        :param x_views: Time series data.
        """
        jittered_x_views = []
        for x_view in x_views:
            noise = np.random.normal(0, self.jitter_std, size=x_view.shape)
            mask = np.random.uniform(0, 1, size=x_view.shape) <= noise_probability
            jittered_x_view = x_view + (mask * noise)
            jittered_x_views.append(jittered_x_view)

        if len(jittered_x_views) == 1:
            jittered_x_views = jittered_x_views[0]
        return jittered_x_views
    
    def time_warp(self, x_view):
        """
        Stretch and squeeze some intervals in the input time series using interpolation.
        :param x_view: Time series data.
        """
        scaling_factor = 1 + np.random.normal(0, self.stretch_std)
        original_length = x_view.shape[-1]
        new_length = original_length
        warped_series = np.zeros((x_view.shape[0], new_length))

        for i in range(x_view.shape[0]):
            # Create a set of new time points based on scaling factor
            new_time_points = np.arange(0, new_length) / scaling_factor
            # Use linear interpolation to fill in the values
            warped_series[i] = np.interp(new_time_points, np.arange(original_length), x_view[i])

        return warped_series