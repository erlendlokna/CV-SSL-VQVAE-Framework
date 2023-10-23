from src.preprocessing.preprocess_ucr import UCRDatasetImporter
from src.preprocessing.data_pipeline import build_data_pipeline
from src.utils import load_yaml_param_settings

from src.models.vqvae_representations import PretrainedVQVAE, BaseVQVAE
from src.experiments.tester import RepTester, plot_results, plot_multiple_results
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

    #regular_vqvae = PretrainedVQVAE(input_length, config)
    #contrastive_vqvae = PretrainedVQVAE(input_length, config, contrastive=True)
    model1 = PretrainedVQVAE(input_length, config, contrastive=False)
    model2 = BaseVQVAE(input_length, config)

    tester1 = RepTester(model1, train_data_loader, test_data_loader, concatenate_zqs=True)
    tester2 = RepTester(model2, train_data_loader, test_data_loader, concatenate_zqs=True)

    test1 = tester1.test_conv2d(n_runs=20)#embed: tests classifiers on PCA and UMAP embeddings
    test2 = tester2.test_conv2d(n_runs=20)
    
    #test = tester.test_flatten(n_runs=20)
    #plot_results(test, embed=False, title=f"{config['dataset']['dataset_name']}(flatten)")
    plot_multiple_results([test1, test2], ["trained", "untrained"])
