from src.models.simple_classification import SVMCodebook, BaseCodeBook
from sklearn.svm import SVC
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
    
    prediction_no_split = SVM.train_and_predict(train_dataloader=train_data_loader, test_dataloader=test_data_loader, kernel='linear')

    prediction_split = SVM.split_dataloader_and_predict(dataloader=test_data_loader, test_size=0.9, kernel='linear')

    print('training on train dataloader and predicting on test dataloader:\n accuracy:', prediction_no_split['accuracy'],'\nconf matrix:', prediction_no_split['confusion_matrix'])
    
    print('splitting test set and training and predicting on this:\n accuracy:', prediction_split['accuracy'], '\nconf matrix:', prediction_split['confusion_matrix'])




    
    