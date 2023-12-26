import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from models.vqvae import VQVAE
from preprocessing.preprocess_ucr import UCRDatasetImporter
from preprocessing.data_pipeline import build_data_pipeline
from utils import load_yaml_param_settings
from utils import save_model

import numpy as np
import torch

torch.set_float32_matmul_precision('medium')

def train_VQVAE(config: dict,
                 train_data_loader: DataLoader,
                 test_data_loader: DataLoader,
                 do_validate: bool,
                 wandb_project_name = '',
                 wandb_project_case_idx: str = '',
                 wandb_run_name=''):
    """
    :param do_validate: if True, validation is conducted during training with a test dataset.
    """
    project_name = wandb_project_name

    if wandb_project_case_idx != '':
        project_name += f'-{wandb_project_case_idx}'

    input_length = train_data_loader.dataset.X.shape[-1]

    train_model = VQVAE(input_length, test_data_loader=test_data_loader, train_data_loader=train_data_loader, config=config, n_train_samples=len(train_data_loader.dataset))


    wandb_logger = WandbLogger(project=project_name, name=wandb_run_name, config=config)
    trainer = pl.Trainer(logger=wandb_logger,
                         enable_checkpointing=False,
                         callbacks=[LearningRateMonitor(logging_interval='epoch')],
                         max_epochs=config['trainer_params']['max_epochs']['vqvae'],
                         devices=config['trainer_params']['gpus'],
                         accelerator='gpu',
                         check_val_every_n_epoch=20)
    
    trainer.fit(train_model,
                train_dataloaders=train_data_loader,
                val_dataloaders=test_data_loader if do_validate else None
                )
    
    # additional log
    n_trainable_params = sum(p.numel() for p in train_model.parameters() if p.requires_grad)
    wandb.log({'n_trainable_params:': n_trainable_params})

    # test
    print('closing...')
    wandb.finish()

    print('saving the models...')
    
    save_model({'vqvae_encoder': train_model.encoder,
                'vqvae_decoder': train_model.decoder,
                'vqvae_vq_model': train_model.vq_model,
                }, id=config['dataset']['dataset_name'])


if __name__ == "__main__":
    config_dir = 'configs/config.yaml' #dir to config file

    config = load_yaml_param_settings(config_dir)

    # data pipeline
    dataset_importer = UCRDatasetImporter(**config['dataset'])
    batch_size = config['dataset']['batch_sizes']['vqvae']
    train_data_loader, test_data_loader = [build_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]

    train_VQVAE(config, train_data_loader, test_data_loader, do_validate=True, wandb_project_name='Barlow-Twins-VQVAE')