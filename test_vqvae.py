import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from src.models.vqvae import VQVAE
from src.preprocessing.preprocess_ucr import UCRDatasetImporter
from src.preprocessing.data_pipeline import build_data_pipeline
from src.utils import load_yaml_param_settings
from src.utils import save_model
from src.models.vqvae import LoadVQVAE


def evaluate_vqvae(config:dict,
                   test_data_loader: DataLoader,
                   wandb_project_case_idx: str = ''):
    project_name = 'RepVQVAE-stage1'
    if wandb_project_case_idx != '':
        project_name += f'-{wandb_project_case_idx}'

    input_length = test_data_loader.dataset.X.shape[-1]

    dataset_name = config['dataset']['dataset_name']

    loaded_vqvae = LoadVQVAE(input_length, config)
    
    wandb_logger = WandbLogger(project=project_name, name=None, config=config)

    trainer = pl.Trainer(logger=wandb_logger,
                         enable_checkpointing=False,
                         callbacks=[LearningRateMonitor(logging_interval='epoch')],
                         max_epochs=config['trainer_params']['max_epochs']['vqvae'],
                         devices=config['trainer_params']['gpus'],
                         accelerator='gpu')
    
    metrics = trainer.test(model=loaded_vqvae, dataloaders = test_data_loader)
    print(metrics)

if __name__ == "__main__":
    config_dir = 'src/configs/config.yaml' #dir to config file

    config = load_yaml_param_settings(config_dir)
    

    # data pipeline
    dataset_importer = UCRDatasetImporter(**config['dataset'])
    batch_size = config['dataset']['batch_sizes']['vqvae']
    _, test_data_loader = [build_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]

    evaluate_vqvae(config, test_data_loader)