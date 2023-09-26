"""
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import wandb


def train_vqvae(config: dict,
                train_data_loader: DataLoader,
                test_data_loader: DataLoader,
                do_validate: bool,
                ):
    
    input_length = train_data_loader.dataset.X.shape[-1]
"""

if __name__ == "__main__":
    from src.models.vqvae import VQVAE
    
