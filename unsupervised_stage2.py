from src.models.simple_classification import KMeansCodeBook, SpectralCodeBook
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


def codebook_classification(config:dict,
                   data_loader: DataLoader,
                   eval=False,
                   num_runs = None):
    
    input_length = data_loader.dataset.X.shape[-1]

    classifier = KMeansCodeBook(input_length, config)
    #classifier = SpectralCodeBook(input_length, config)
    
    
    if eval: #run: "python stage2.py eval 10" in terminal to run eval with 10 runs
        accuracies = np.zeros(num_runs)
        #running num_runs number of classifications and grabbing accuracy
        for i in tqdm(range(num_runs)):
            cluster_data = classifier.classify(data_loader)
            accuracies[i] = cluster_data['accuracy']

        mean_acc = np.mean(accuracies)

        f, ax = plt.subplots()
        ax.set_title(f"Accuracy and SSE for {classifier.classifier_name}, {num_runs} runs, dataset: {config['dataset']['dataset_name']}")
        x = np.arange(num_runs)
        ax.set_ylabel("Accuracy")
        ax.plot(x, accuracies, c="tab:blue")
        ax.plot(x, [mean_acc for _ in range(num_runs)], '--', c="grey", label=f"mean acc: {mean_acc}")
        ax.legend()

        f_hist, ax_hist = plt.subplots()
        ax_hist.hist(accuracies, bins=round(0.3 * num_runs))
        ax_hist.set_title(f"Histogram for {classifier.classifier_name}, {num_runs} runs, dataset: {config['dataset']['dataset_name']}")

        f.savefig(f"results/{num_runs}_{classifier.classifier_name}_{config['dataset']['dataset_name']}.png")
        f_hist.savefig(f"results/{num_runs}_{classifier.classifier_name}_{config['dataset']['dataset_name']}_hist.png")
    else:
        cluster_data = classifier.classify(data_loader)
        print(cluster_data['accuracy'])

if __name__ == "__main__":
    config_dir = 'src/configs/config.yaml' #dir to config file

    config = load_yaml_param_settings(config_dir)
    
    # data pipeline
    dataset_importer = UCRDatasetImporter(**config['dataset'])
    batch_size = config['dataset']['batch_sizes']['vqvae']
    train_data_loader, test_data_loader = [build_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]

    codebook_classification(config, test_data_loader, eval='eval' in sys.argv, num_runs=int(sys.argv[2]))