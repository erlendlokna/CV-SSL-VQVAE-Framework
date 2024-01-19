from preprocessing.preprocess_ucr import UCRDatasetImporter
from preprocessing.data_pipeline import build_data_pipeline
from utils import load_yaml_param_settings
from utils import save_model
import torch

torch.set_float32_matmul_precision('medium')

from train_BTVQVAE import train_BTVQVAE

from train_VQVAE import train_VQVAE

n_runs = 2

UCR_subset = [
    'StarLightCurves',
    'ElectricDevices',
    'ECG5000',
    'Wafer',
    'TwoPatterns',
    'ShapesAll',
    'FordA',
    'UWaveGestureLibraryAll',
    'ChlorineConcentration',
    'FordB',
    'StarLightCurves',
    'ElectricDevices',
    'ECG5000',
    'Wafer',
    'TwoPatterns',
    'ShapesAll'
    "CBF",
]
finished_vqvae = [

]

finished_barlow = [

]

all_augs = ['AmpR','STFT', 'jitter', 'slope', 'flip']

gammas = [2, 1, 0.5]

wandb_project_name = "BarlowTwinsVQVAE"

def update_config(config, beta, dataset):
    c = config
    c['dataset']['dataset_name'] = dataset
    c['barlow_twins']['beta'] = beta
    return c

run_name_barlow = lambda dataset, gamma, run: f"BVQVAE_{dataset}_allaugs_gamma_{gamma}_run_{run}"
run_name_vqvae = lambda dataset, run: f"VQVAE_{dataset}_run_{run}"

if __name__ == "__main__":
    config_dir = 'configs/config.yaml' #dir to config file

    config = load_yaml_param_settings(config_dir)

    for run in range(n_runs):
        print(f"Run {run}")
        for ucr_dataset in UCR_subset:
            print(f"dataset: {ucr_dataset}")

            config = update_config(config, gammas[0], ucr_dataset)

            # data pipeline
            dataset_importer = UCRDatasetImporter(**config['dataset'])
            batch_size = config['dataset']['batch_sizes']['vqvae']
            train_data_loader_non_aug, test_data_loader= [build_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]
            
            #augmented data pipeline:
            augmentations = all_augs 
            train_data_loader_aug = build_data_pipeline(batch_size, dataset_importer, config, "train", augmentations)

            #running vqvae experiment
            if ucr_dataset not in finished_vqvae:
                train_VQVAE(config, train_data_loader_non_aug, test_data_loader, 
                            wandb_project_name=wandb_project_name, 
                            wandb_run_name=run_name_vqvae(ucr_dataset, run),
                            do_validate=True)

            for gamma in gammas:
                #overwriting config:

                config = update_config(config, gamma, ucr_dataset)
                
                #running Barlow VQVAE experiment
                if [ucr_dataset, gamma] not in finished_barlow:
                    train_BTVQVAE(config, aug_train_data_loader = train_data_loader_aug,
                                train_data_loader=train_data_loader_non_aug,
                                test_data_loader=test_data_loader, 
                                wandb_project_name=wandb_project_name,
                                wandb_run_name=run_name_barlow(ucr_dataset, gamma, run),
                                do_validate=True)
                    
            