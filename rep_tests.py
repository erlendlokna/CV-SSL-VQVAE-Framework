from src.preprocessing.preprocess_ucr import UCRDatasetImporter
from src.preprocessing.data_pipeline import build_data_pipeline
from src.utils import load_yaml_param_settings

from src.models.vqvae_representations import PretrainedVQVAE, BaseVQVAE
from src.experiments.tester import RepTester, plot_results, plot_multiple_results, UMAP_plots, PCA_plots
from src.experiments.supervised_tests import supervised_test

import numpy as np


if __name__ == "__main__": 
    import sys
    # ---- Config -----
    config_dir = 'src/configs/config.yaml' #dir to config file

    config = load_yaml_param_settings(config_dir)

    if len(sys.argv) > 1: config['dataset']['dataset_name'] = sys.argv[1]

    # ---- Getting data ----
    dataset_importer = UCRDatasetImporter(**config['dataset'])
    batch_size = config['dataset']['batch_sizes']['vqvae']
    train_data_loader, test_data_loader = [build_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]

    input_length = train_data_loader.dataset.X.shape[-1]

    model1 = PretrainedVQVAE(input_length, config)
    model2 = BaseVQVAE(input_length, config)
    #validation:
    print("Validation of VQVAEs:")
    print("MEA on test_data_loader:", np.mean(model1.validate(test_data_loader)))
    print("MEA on test_data_loader:", np.mean(model2.validate(test_data_loader)))

    print("Performing representation tests:")
    tester1 = RepTester(model1, train_data_loader, test_data_loader, concatenate_zqs=True)
    tester2 = RepTester(model2, train_data_loader, test_data_loader, concatenate_zqs=True)
    
    
    print("Intrinstic dimensions:")
    print("trained:",tester1.test_intristic_dimension())
    print("untrained", tester2.test_intristic_dimension())


    test1 = tester1.test_flatten_zqs(n_runs=20)#embed: tests classifiers on PCA and UMAP embeddings
    test2 = tester2.test_flatten_zqs(n_runs=20)

    print("PCA plots:")
    PCA_plots(
        [tester1.flatten_zqs(), tester2.flatten_zqs()],
        [tester1.get_y(), tester2.get_y()],
        ["trained", "non-trained"]
    )

    #plot_results(test, embed=False, title=f"{config['dataset']['dataset_name']}(flatten)")
    plot_multiple_results([test1, test2], ["trained", "untrained"])
