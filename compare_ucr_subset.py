import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from src.models.BarlowTwinsVQVAE import BarlowTwinsVQVAE

from src.preprocessing.preprocess_ucr import UCRDatasetImporter
from src.preprocessing.data_pipeline import build_data_pipeline
from src.utils import load_yaml_param_settings
from src.utils import save_model
import torch
from plotting import sample_plot_classes

torch.set_float32_matmul_precision('medium')

from train_barlowvqvae import train_BarlowVQVAE

from train_vqvae import train_VQVAE

experiment_name = "STFT-jitter"

UCR_subset = [
    'StarLightCurves'
    'ElectricDevices'
    'StarLightCurves',
    'ECG5000',
    'Wafer',
    'TwoPatters',
    'ShapesAll'
]

all_augs = ['AmpR','STFT', 'jitter', 'slope', 'flip']

betas = [1, 0.6, 0.3, 0.1]

epochs = 1000

finished_barlow_datasets = [
]

finished_vqvae_datasets = [
]

betas = [
    1, 0.6, 0.3, 0.1
]

all_augs = ['AmpR','STFT', 'jitter', 'slope', 'flip']

epochs = 1000



wandb_project_name = "Barlow-Twins-VQVAE"

if __name__ == "__main__":
    config_dir = 'src/configs/config.yaml' #dir to config file

    config = load_yaml_param_settings(config_dir)

    for ucr_dataset in UCR_subset:
        config['dataset']['dataset_name'] = ucr_dataset

        # data pipeline
        dataset_importer = UCRDatasetImporter(**config['dataset'])
        batch_size = config['dataset']['batch_sizes']['vqvae']
        train_data_loader_non_aug, test_data_loader= [build_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]

        augmentations = ['AmpR','STFT', 'jitter', 'slope']
        train_data_loader_aug = build_data_pipeline(batch_size, dataset_importer, config, "train", augmentations)
        train_data_loader_not_aug = build_data_pipeline(batch_size, dataset_importer, config, "train")

        #experiments:
        if ucr_dataset not in finished_vqvae_datasets:
            train_VQVAE(config, train_data_loader_not_aug, test_data_loader, wandb_project_name=wandb_project_name, do_validate=True)


        if ucr_dataset not in finished_barlow_datasets:
            train_BarlowVQVAE(config, aug_train_data_loader = train_data_loader_aug,
                        train_data_loader=train_data_loader_non_aug,
                        test_data_loader=test_data_loader, 
                        wandb_project_name=wandb_project_name,
                        wandb_exp_name=experiment_name,
                        do_validate=True)
            
        