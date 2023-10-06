from src.models.simple_classification import KMeansCodeBook
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from src.models.vqvae import VQVAE
from src.preprocessing.preprocess_ucr import UCRDatasetImporter
from src.preprocessing.data_pipeline import build_data_pipeline
from src.utils import load_yaml_param_settings
from src.utils import save_model
from src.models.vqvae import LoadVQVAE


def codebook_classification(config:dict,
                   data_loader: DataLoader,
                   eval=True):
    
    input_length = data_loader.dataset.X.shape[-1]

    classifier = KMeansCodeBook(input_length, config)

    if eval:
        num_runs = 500
        accuracies, sse = classifier.multiple_classifications(data_loader, num=num_runs)
        mean_acc = np.mean(accuracies)
        mean_sse = np.mean(sse)

        f, ax1 = plt.subplots()
        ax1.set_title(f"Accuracy and SSE for {classifier.classifier_name}, {num_runs} runs, dataset: {config['dataset']['dataset_name']}")
        x = np.arange(num_runs)
        ax1.set_ylabel("Accuracy")
        ax1.plot(x, accuracies, c="tab:blue")
        ax1.plot(x, [mean_acc for _ in range(num_runs)], '--', c="grey", label=f"mean acc: {mean_acc}")
        ax1.legend()

        ax2 = ax1.twinx()
        ax2.set_ylabel("SSE")
        ax2.plot(x, sse, c="tab:red")
        ax2.plot(x, [mean_sse for _ in range(num_runs)], '--', c="grey", label=f"mean sse: {mean_sse}")
        ax2.legend()

        f_hist, ax_hist = plt.subplots()
        ax_hist.hist(accuracies, bins=50)
        ax_hist.set_title(f"Histogram for {classifier.classifier_name}, {num_runs} runs, dataset: {config['dataset']['dataset_name']}")

        f.savefig(f"{num_runs}_{classifier.classifier_name}_{config['dataset']['dataset_name']}.png")
        f_hist.savefig(f"{num_runs}_{classifier.classifier_name}_{config['dataset']['dataset_name']}_hist.png")
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

    codebook_classification(config, test_data_loader, eval=False)