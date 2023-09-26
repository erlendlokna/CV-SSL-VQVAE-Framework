from src.models.vqvae import VQVAE
from src.utils import load_yaml_param_settings
from src.preprocessing.data_pipeline import build_data_pipeline
from src.preprocessing.preprocess_ucr import UCRDatasetImporter


if __name__ == "__main__":
    config_dir = 'src/configs/config.yaml' #dir to config file

    config = load_yaml_param_settings(config_dir)

    # data pipeline
    dataset_importer = UCRDatasetImporter(**config['dataset'])
    batch_size = config['dataset']['batch_sizes']['vqvae']
    train_data_loader, test_data_loader = [build_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]

    input_length = train_data_loader.dataset.X.shape[-1]
    train_exp = VQVAE(input_length, config, len(train_data_loader.dataset))
    
    x, y = dataset_importer.X_train[0], dataset_importer.Y_train[0]
    batch = (x, y)
    VQVAE.forward(batch)
