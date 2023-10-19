import numpy as np
from einops import rearrange
import torch
import yaml
import os
from pathlib import Path
import tempfile
import requests
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA
from sklearn import metrics

def get_root_dir():
    return Path(__file__).parent.parent.parent

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


def save_model(models_dict: dict, dirname='saved_models', id: str = ''):
    """
    :param models_dict: {'model_name': model, ...}
    """
    try:
        if not os.path.isdir(get_root_dir().joinpath(dirname)):
            os.mkdir(get_root_dir().joinpath(dirname))

        id_ = id[:]
        if id != '':
            id_ = '-' + id_
        for model_name, model in models_dict.items():
            torch.save(model.state_dict(), get_root_dir().joinpath(dirname, model_name + id_ + '.ckpt'))
    except PermissionError:
        # dirname = tempfile.mkdtemp()
        dirname = tempfile.gettempdir()
        print(f'\nThe trained model is saved in the following temporary dirname due to some permission error: {dirname}.\n')

        id_ = id[:]
        if id != '':
            id_ = '-' + id_
        for model_name, model in models_dict.items():
            torch.save(model.state_dict(), get_root_dir().joinpath(dirname, model_name + id_ + '.ckpt'))


def download_ucr_datasets(url='https://figshare.com/ndownloader/files/37909926', chunk_size=128, zip_fname='UCR_archive.zip'):
    #dirname = str(get_root_dir().joinpath("data"))
    dirname = './data'
    if os.path.isdir(os.path.join(dirname, 'UCRArchive_2018')):
        return None

    if not os.path.isdir(dirname) or not os.path.isfile(os.path.join(dirname, zip_fname)):
        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        # download
        r = requests.get(url, stream=True)
        print('downloading the UCR archive datasets...\n')
        fname = os.path.join(dirname, zip_fname)
        with open(fname, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=chunk_size):
                fd.write(chunk)

        # unzip
        with tarfile.open(fname) as ff:
            ff.extractall(path=dirname)
    elif os.path.isfile(str(get_root_dir().joinpath("datasets", zip_fname))):
        # unzip
        fname = os.path.join(dirname, zip_fname)
        with tarfile.open(fname) as ff:
            ff.extractall(path=dirname)
    else:
        pass

    os.remove('./data/UCR_archive.zip')


class UMAP_wrapper:
    def __init__(self, model, n_comps=2): 
        self.model = model
        self.reducer = umap.UMAP(n_components=n_comps)

    def fit(self, x_train, y_labs):
        train_embs = self.reducer.fit_transform(x_train)
        self.model.fit(train_embs, y_labs)

    def predict(self, x_test):
        test_embs = self.reducer.transform(x_test)
        return self.model.predict(test_embs)

class PCA_wrapper:
    def __init__(self, model, var_explained_crit=0.9): 
        self.model = model
        self.pca = PCA(var_explained_crit)

    def fit(self, x_train, y_labs):
        train_embs = self.pca.fit_transform(x_train)
        self.model.fit(train_embs, y_labs)

    def predict(self, x_test):
        test_embs = self.pca.transform(x_test)
        preds = self.model.predict(test_embs)
        return preds
    
def UMAP_plots(X, labs, labs2=None, comps = 2):
    embs = umap.UMAP(densmap=True, n_components=comps, random_state=42).fit(X).embedding_

    if labs2 is not None:
        f, ax = plt.subplots(1, 2, figsize=(16,8))
        ax[0].scatter(embs[:, 0], embs[:, 1], c=labs); ax[0].set_title("true")
        ax[1].scatter(embs[:, 0], embs[:, 1], c=labs2); ax[1].set_title("predicted")
        plt.show()
    else:
        f, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(embs[:, 0], embs[:, 1], c = labs);
        plt.show()


def unsupervised_score(labs_true, labs_pred):
    print(f"rand score: {metrics.rand_score(labs_true, labs_pred)}")
    print(f"adjusted rand score: {metrics.adjusted_rand_score(labs_true, labs_pred)}")
    print(f"normalized mutual info score: {metrics.normalized_mutual_info_score(labs_true, labs_pred)}")