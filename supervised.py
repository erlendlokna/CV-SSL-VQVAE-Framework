from src.models.classification import SVM_VQVAE, evaluate_classifier
import wandb

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
    
    argv = sys.argv #terminal arguments


    svm_no_split = SVM_VQVAE(input_length, config,
                            train_data_loader=train_data_loader)
    
    svm_split_on_train = SVM_VQVAE(input_length, config,
                                   train_data_loader=train_data_loader,
                                   test_size=0.2)
    
    svm_split_on_test = SVM_VQVAE(input_length, config,
                                  train_data_loader=test_data_loader,
                                  test_size=0.2)
    

    if argv[1] == 'eval': #argv[0] = 'supervised.py'
        num_test = int(argv[2]) 
        accs1, mean_acc1 = evaluate_classifier(svm_no_split, test_data_loader=test_data_loader, num_tests=num_test)
        accs2, mean_acc2 = evaluate_classifier(svm_split_on_train, num_tests=num_test)
        accs3, mean_acc3 = evaluate_classifier(svm_split_on_test, num_tests=num_test)
        
        fix, ax = plt.subplots()
        x = np.arange(num_test)
        ax.plot(x, accs1, label="No split")
        ax.plot(x, [mean_acc1 for _ in range(num_test)], '--', c='grey')
        ax.plot(x, accs2, label="Split on train")
        ax.plot(x, [mean_acc2 for _ in range(num_test)], '--', c='grey')
        ax.plot(x, accs3, label="Split on test")
        ax.plot(x, [mean_acc3 for _ in range(num_test)], '--', c='grey')
        ax.legend()
        plt.show()


    
    