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

UCR_subset = [
    'StarLightCurves',
    'ElectricDevices',
    'ECG5000',
    'Wafer',
    'TwoPatterns',
    'ShapesAll',
    'FordA',
    'UWaveGestureLibraryAll',
    'ChlorineConcentration',
    'FordB'

]

all_augs = ['AmpR','STFT', 'jitter', 'slope', 'flip']

betas = [2, 1, 0.5]

finished_barlow_datasets = [
    ["StarLightCurves", 2],
    ["StarLightCurves", 1],
    ["StarLightCurves", 0.5],
    ["ElectricDevices", 2],
    ["ElectricDevices", 1]
]

finished_vqvae_datasets = [
    "StarLightCurves",
    "ElectricDevices"
]

wandb_project_name = "BarlowTwinsVQVAE"


def update_config(config, beta, dataset):
    c = config
    c['dataset']['dataset_name'] = dataset
    c['barlow_twins']['beta'] = beta
    return c

run_name_barlow = lambda dataset, beta: f"BVQVAE_{dataset}_allaugs_beta_{beta}_run2"
run_name_vqvae = lambda dataset: f"VQVAE_{dataset}_run2"

if __name__ == "__main__":
    config_dir = 'src/configs/config.yaml' #dir to config file

    config = load_yaml_param_settings(config_dir)

    for ucr_dataset in UCR_subset:

        config = update_config(config, betas[0], ucr_dataset)

        # data pipeline
        dataset_importer = UCRDatasetImporter(**config['dataset'])
        batch_size = config['dataset']['batch_sizes']['vqvae']
        train_data_loader_non_aug, test_data_loader= [build_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]
        
        #augmented data pipeline:
        augmentations = all_augs 
        train_data_loader_aug = build_data_pipeline(batch_size, dataset_importer, config, "train", augmentations)

        #running vqvae experiment
        if ucr_dataset not in finished_vqvae_datasets:
            train_VQVAE(config, train_data_loader_non_aug, test_data_loader, 
                        wandb_project_name=wandb_project_name, 
                        wandb_run_name=run_name_vqvae(ucr_dataset),
                        do_validate=True)

        for beta in betas:
            #overwriting config:
            print(f"Training BVQVAE for {ucr_dataset} and beta={beta}")
            config = update_config(config, beta, ucr_dataset)
            
            #running Barlow VQVAE experiment
            if [ucr_dataset, beta] not in finished_barlow_datasets:
            
                train_BarlowVQVAE(config, aug_train_data_loader = train_data_loader_aug,
                            train_data_loader=train_data_loader_non_aug,
                            test_data_loader=test_data_loader, 
                            wandb_project_name=wandb_project_name,
                            wandb_run_name=run_name_barlow(ucr_dataset,beta),
                            do_validate=True)
                
            