from src.models.vqvae import VQVAE
from src.utils import load_yaml_param_settings
from src.preprocessing.data_pipeline import build_data_pipeline
from src.preprocessing.preprocess_ucr import UCRDatasetImporter
from train_vqvae import train_VQVAE

if __name__ == "__main__":
    
    config_dir = 'src/configs/config.yaml' #dir to config file

    config = load_yaml_param_settings(config_dir)

    # data pipeline
    dataset_importer = UCRDatasetImporter(**config['dataset'])
    batch_size = config['dataset']['batch_sizes']['vqvae']
    train_data_loader, test_data_loader = [build_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]

    #input_length = train_data_loader.dataset.X.shape[-1]
    #train_exp = VQVAE(input_length, config, len(train_data_loader.dataset))
    
    train_VQVAE(config, train_data_loader, test_data_loader, do_validate=False)
    
