from src.utils import load_yaml_param_settings
from src.preprocessing.preprocess_ucr import UCRDatasetImporter
import matplotlib.pyplot as plt
import numpy as np
import random
from plotting import sample_plot_classes

def view_data():
    import os


    dirs = os.listdir("data/UCRArchive_2018")
    
    for name in dirs:
        config_dir = 'src/configs/config.yaml' #dir to config file
        config = load_yaml_param_settings(config_dir)
        config['dataset']['dataset_name'] = name
        dataset_importer = UCRDatasetImporter(**config['dataset'])

        training_data = dataset_importer.X_train
        test_data = dataset_importer.X_test

        labels = dataset_importer.Y_train
        name = config['dataset']['dataset_name']

        sample_plot_classes(training_data, labels, name)


if __name__ == "__main__":
    view_data()