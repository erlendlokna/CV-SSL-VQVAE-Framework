from src.preprocessing.preprocess_ucr import UCRDatasetImporter
from src.preprocessing.data_pipeline import build_data_pipeline
from src.utils import load_yaml_param_settings

from src.experiments.tester import RepTester, plot_results
from src.experiments.supervised_tests import supervised_test

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

    tester = RepTester(config, input_length, train_data_loader, test_data_loader)

    test = tester.test_flatten(n_runs=20, embed=False)
    plot_results(test, embed=False, title=f"{config['dataset']['dataset_name']}(flatten)")

