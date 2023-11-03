import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from src.models.con_vqvae import InfoNCE_VQVAE
from src.models.con_vqvae import BarlowTwinsVQVAE

from src.preprocessing.augmentations import Augmentations
from src.preprocessing.preprocess_ucr import AugUCRDataset, UCRDataset, UCRDatasetImporter
from src.preprocessing.data_pipeline import build_data_pipeline
from src.utils import load_yaml_param_settings
from src.utils import save_model

import numpy as np

def train_ConVQVAE(config: dict,
                aug_train_data_loader: DataLoader,
                aug_test_data_loader: DataLoader,
                train_data_loader: DataLoader,
                test_data_loader: DataLoader,
                do_validate: bool,
                wandb_project_case_idx: str = ''):
    """
    :param do_validate: if True, validation is conducted during training with a test dataset.
    """
    project_name = 'RepVQVAE-stage1'
    if wandb_project_case_idx != '':
        project_name += f'-{wandb_project_case_idx}'

    input_length = train_data_loader.dataset.X.shape[-1]

    train_model = BarlowTwinsVQVAE(input_length, non_aug_test_data_loader=test_data_loader,
                                    non_aug_train_data_loader=train_data_loader, 
                                    config=config, n_train_samples=len(train_data_loader.dataset))

    wandb_logger = WandbLogger(project=project_name, name=None, config=config)
    trainer = pl.Trainer(logger=wandb_logger,
                         enable_checkpointing=False,
                         callbacks=[LearningRateMonitor(logging_interval='epoch')],
                         max_epochs=config['trainer_params']['max_epochs']['vqvae'],
                         devices=config['trainer_params']['gpus'],
                         accelerator='gpu',
                         check_val_every_n_epoch=20)
    
    trainer.fit(train_model,
                train_dataloaders=aug_train_data_loader,
                val_dataloaders=aug_test_data_loader if do_validate else None
                )
    
    # additional log
    n_trainable_params = sum(p.numel() for p in train_model.parameters() if p.requires_grad)
    wandb.log({'n_trainable_params:': n_trainable_params})

    # test
    print('closing...')
    wandb.finish()

    print('saving the models...')
    
    
    save_model({'contrastive_encoder': train_model.encoder,
                'contrastive_decoder': train_model.decoder,
                'contrastive_vq_model': train_model.vq_model,
                }, id=config['dataset']['dataset_name'])
    
    
if __name__ == "__main__":
    config_dir = 'src/configs/config.yaml' #dir to config file

    config = load_yaml_param_settings(config_dir)

    # data pipeline
    dataset_importer = UCRDatasetImporter(**config['dataset'])
    batch_size = config['dataset']['batch_sizes']['vqvae']
    train_data_loader_non_aug, test_data_loader= [build_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]

    augmentations = ['AmpR']
    train_data_loader_aug = build_data_pipeline(batch_size, dataset_importer, config, "train", augmentations)

    train_ConVQVAE(config, aug_test_data_loader=train_data_loader_aug,
                   aug_train_data_loader= test_data_loader,
                    train_data_loader=train_data_loader_non_aug,
                    test_data_loader=test_data_loader, do_validate=False)