import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from src.models.vqvae import VQVAE
from src.preprocessing.preprocess_ucr import UCRDatasetImporter
from src.preprocessing.data_pipeline import build_data_pipeline
from src.utils import save_model


def train_VQVAE(config: dict,
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
    train_model = VQVAE(input_length, config, len(train_data_loader.dataset))
    wandb_logger = WandbLogger(project=project_name, name=None, config=config)
    trainer = pl.Trainer(logger=wandb_logger,
                         enable_checkpointing=False,
                         callbacks=[LearningRateMonitor(logging_interval='epoch')],
                         max_epochs=config['trainer_params']['max_epochs']['vqvae'],
                         devices=config['trainer_params']['gpus'],
                         accelerator='gpu')
    
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
    save_model({'encoder_l': train_model.encoder,
                'decoder_l': train_model.decoder,
                'vq_model_l': train_model.vq_model,
                }, id=config['dataset']['dataset_name'])
