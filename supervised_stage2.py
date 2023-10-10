from src.models.simple_classification import SVMCodebook
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import sys

from src.models.vqvae import VQVAE
from src.preprocessing.preprocess_ucr import UCRDatasetImporter
from src.preprocessing.data_pipeline import build_data_pipeline
from src.utils import load_yaml_param_settings
from src.utils import save_model
from src.models.vqvae import LoadVQVAE


if __name__ == "__main__":
    config_dir = 'src/configs/config.yaml' #dir to config file

    config = load_yaml_param_settings(config_dir)
    
    # data pipeline
    dataset_importer = UCRDatasetImporter(**config['dataset'])
    batch_size = config['dataset']['batch_sizes']['vqvae']
    train_data_loader, test_data_loader = [build_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]

    input_length = train_data_loader.dataset.X.shape[-1]
    

    
    SVM = SVMCodebook(input_length, config)
    
    #SVM.train(train_data_loader, kernel='linear')
    
    #prediction_data = SVM.classify(test_data_loader)
    

    #prediction_data = SVM.split_and_classify(test_data_loader, kernel='linear')
    prediction_data = SVM.split_and_classify(test_data_loader, kernel='sigmoid')
    print(prediction_data['accuracy'])

    

    
    