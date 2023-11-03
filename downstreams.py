from src.preprocessing.preprocess_ucr import UCRDatasetImporter
from src.preprocessing.data_pipeline import build_data_pipeline
from src.utils import load_yaml_param_settings

from src.preprocessing.preprocess_ucr import UCRDataset
from discrete_latents import PretrainedLatents, RandomInitLatents
from src.experiments.datahandler import LatentDataHandler
from src.experiments.tests import (knn_test, svm_test,
                                intristic_dimension, classnet_test,
                                multiple_tests, plot_tests,
                                pca_plots, umap_plots, tsne_plot)
import sys
import numpy as np

def probe_tests(pooler, knn=False, svm=False, classnet=False, cnnclassnet=False, n_runs = 10, n_epochs = 200, concatenate = True):
    results = []
    labels = []
    if knn: 
        results.append(multiple_tests(knn_test, pooler.zqs(flatten=True), pooler.get_y(), n_runs= n_runs))
        labels.append("KNN")
    if svm: 
        results.append(multiple_tests(svm_test, pooler.zqs(flatten=True), pooler.get_y(), n_runs= n_runs))
        labels.append("SVM")
    if classnet: 
        results.append(multiple_tests(classnet_test, pooler.zqs(flatten=True), pooler.get_y(), n_runs = n_runs, num_epochs=n_epochs))
        labels.append("Neural Net")
    if cnnclassnet: 
        results.append(multiple_tests(classnet_test, pooler.zqs(flatten=False), pooler.get_y(), n_runs = n_runs, num_epochs=n_epochs, CNN=True))
        labels.append("CNN Neural Net")
    
    return results, labels

def test_representations():
     # ---- Config -----
    config_dir = 'src/configs/config.yaml' #dir to config file

    config = load_yaml_param_settings(config_dir)

    if len(sys.argv) > 1: config['dataset']['dataset_name'] = sys.argv[1]

    # ---- Getting data ----
    dataset_importer = UCRDatasetImporter(**config['dataset'])
    batch_size = config['dataset']['batch_sizes']['vqvae']
    train_data_loader, test_data_loader = [build_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]

    input_length = train_data_loader.dataset.X.shape[-1]

    #model1 = PretrainedLatents(input_length, config, contrastive=True)
    model2 = PretrainedLatents(input_length, config)
    model3 = RandomInitLatents(input_length, config)

    print("Validation of VQVAEs...")
    #print("1) MEA on test_data_loader:", np.mean(model1.validate(test_data_loader, vizualise=False)))
    print("2) MEA on test_data_loader:", np.mean(model2.validate(test_data_loader, vizualise=False)))
    print("3) MEA on test_data_loader:", np.mean(model3.validate(test_data_loader, vizualise=False)))
    
    #latent1 = LatentDataHandler(model1, train_data_loader, test_data_loader, concatenate_zqs=True)
    latent2 = LatentDataHandler(model2, train_data_loader, test_data_loader, concatenate_zqs=True)
    latent3 = LatentDataHandler(model3, train_data_loader, test_data_loader, concatenate_zqs=True)

    print("Intristic dimensions...")
    #print(f"1) {intristic_dimension(latent1.zqs(flatten=True))}")
    print(f"2) {intristic_dimension(latent2.zqs(flatten=True))}")
    print(f"3) {intristic_dimension(latent3.zqs(flatten=True))}")

    print("Testing probes...")
    #results, labels = probe_tests(pooler3, knn=True, svm=True, classnet=True, cnnclassnet=True, n_runs=100)
    #plot_tests(results, labels)

    print("Plots...")
    #tsne_plot(latent1.zqs(flatten=True), latent1.get_y())
    tsne_plot(latent2.zqs(flatten=True), latent2.get_y())
    tsne_plot(latent3.zqs(flatten=True), latent3.get_y())


if __name__ == "__main__": 
    test_representations()