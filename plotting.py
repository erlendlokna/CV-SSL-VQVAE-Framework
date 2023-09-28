from src.utils import load_yaml_param_settings
from src.preprocessing.preprocess_ucr import UCRDatasetImporter
import matplotlib.pyplot as plt
import numpy as np



def sample_plot_classes(data, labels):
    """
    Displays a "n x 3" grid of plots, where n is the number of classes.
    """
    # Calculate the number of unique labels
    nr_unique_labels = len(np.unique(labels))

    # Define a colormap based on the number of unique labels
    colormap = plt.cm.get_cmap('viridis', nr_unique_labels)  # You can choose a different colormap

    figure = plt.figure(figsize=(12, 8))
    cols, rows = 3, nr_unique_labels

    for i, label in enumerate(np.unique(labels)):
        data_indices = np.where(labels == label)[0]
        for j in range(1, cols + 1):
            if j <= len(data_indices):
                sample_idx = data_indices[j - 1]
                X = data[sample_idx]
                figure.add_subplot(rows, cols, i * cols + j)
                plt.title(f'Label {label}')
                plt.plot(X, color=colormap(label / (nr_unique_labels - 1)))  # Normalize label for colormap

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    #Get dataset
    config_dir = 'src/configs/config.yaml' #dir to config file
    config = load_yaml_param_settings(config_dir)
    dataset_importer = UCRDatasetImporter(**config['dataset'])

    training_data = dataset_importer.X_train
    labels = dataset_importer.Y_train


    sample_plot_classes(training_data, labels)

