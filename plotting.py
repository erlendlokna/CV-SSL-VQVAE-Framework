from src.utils import load_yaml_param_settings
from src.preprocessing.preprocess_ucr import UCRDatasetImporter
import matplotlib.pyplot as plt
import numpy as np
import random



def sample_plot_classes(data, labels, name):
    """
    Displays a "n x 3" grid of plots, where n is the number of classes,
    and returns the figure object for further processing or saving.
    """
    # Calculate the number of unique labels
    nr_unique_labels = len(np.unique(labels))

    # Define a colormap based on the number of unique labels
    colormap = plt.cm.get_cmap('viridis', nr_unique_labels)  

    # Create a figure object to be returned
    figure = plt.figure(figsize=(12, 8))
    cols, rows = 3, nr_unique_labels

    for i, label in enumerate(np.unique(labels)):
        data_indices = np.where(labels == label)[0]
        nr_of_samples = np.min([cols, len(data_indices)])
        sample_idx = random.sample(list(data_indices), nr_of_samples)
        for j in range(nr_of_samples):
            X = data[sample_idx[j]]
            ax = figure.add_subplot(rows, cols, i * cols + j + 1)
            ax.plot(X, color=colormap(label / (nr_unique_labels - 1)))  # Normalize label for colormap
            ax.set_title(f'Label {label}')

    plt.tight_layout()
    plt.suptitle(f'Dataset: {name}', x=0.1, y=0.95, fontsize=16)

    # Instead of plt.show(), we return the figure object
    return figure

def sample_plot_classes2(data, labels, name):
    """
    Displays a "1 x #labels" grid of plots, where #labels is the number of unique classes,
    and returns the figure object for further processing or saving.
    """
    # Calculate the number of unique labels
    nr_unique_labels = len(np.unique(labels))

    # Define a colormap based on the number of unique labels
    colormap = plt.cm.get_cmap('viridis', nr_unique_labels)  

    # Create a figure object to be returned
    figure = plt.figure(figsize=(3 * nr_unique_labels, 3))

    for i, label in enumerate(np.unique(labels)):
        data_indices = np.where(labels == label)[0]
        sample_idx = random.choice(data_indices)
        X = data[sample_idx]
        ax = figure.add_subplot(1, nr_unique_labels, i + 1)
        ax.plot(X, color=colormap(label / (nr_unique_labels - 1)))  # Normalize label for colormap
        ax.set_title(f'Label {label}')

    plt.tight_layout()
    plt.suptitle(f'Dataset: {name}', x=0.5, y=1.05, fontsize=16)

    # Instead of plt.show(), we return the figure object
    return figure


if __name__ == "__main__":

    #Get dataset
    config_dir = 'src/configs/config.yaml' #dir to config file
    config = load_yaml_param_settings(config_dir)

    import sys
    args = sys.argv
    if len(args) > 1:
        config['dataset']['dataset_name'] = args[1]

    dataset_importer = UCRDatasetImporter(**config['dataset'])

    training_data = dataset_importer.X_train
    test_data = dataset_importer.X_test

    labels = dataset_importer.Y_train
    name = config['dataset']['dataset_name']


    f = sample_plot_classes2(training_data, labels, name)
    plt.savefig("dataset.png")
        # sample_plot_classes(test_data, labels, name)
    

