if __name__ == "__main__":
    from src.models.con_vqvae import BarlowTwinsVQVAE
    from src.preprocessing.data_pipeline import build_data_pipeline
    from src.utils import load_yaml_param_settings
    from src.models.con_vqvae import BarlowTwinsVQVAE
    from src.models.vqvae import VQVAE
    from src.preprocessing.preprocess_ucr import Augmentations, AugUCRDataset, UCRDataset, UCRDatasetImporter
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import torch
    import numpy as np
    config_dir = 'src/configs/config.yaml' #dir to config file

    config = load_yaml_param_settings(config_dir)

    # data pipeline
    dataset_importer = UCRDatasetImporter(**config['dataset'])
    batch_size = config['dataset']['batch_sizes']['vqvae']
    train_data_loader_non_aug, test_data_loader= [build_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]

    input_length = train_data_loader_non_aug.dataset.X.shape[-1]
    augmentations = ['AmpR']
    train_data_loader_aug = build_data_pipeline(batch_size, dataset_importer, config, "train", augmentations)

    m = BarlowTwinsVQVAE(input_length, test_data_loader, train_data_loader_non_aug, config, len(train_data_loader_non_aug.dataset))
    m_reg = VQVAE(input_length, test_data_loader, train_data_loader_non_aug, config, len(train_data_loader_non_aug.dataset))
    
    for batch in train_data_loader_non_aug:
        m_reg(batch)

    i = 0
    for batch in train_data_loader_aug:
        subxs_pairs, y = batch
        random_pair_index = np.random.randint(0, len(subxs_pairs))
        x1, x2 = subxs_pairs[random_pair_index]

        f, a = plt.subplots()
        a.plot(torch.flatten(x1[0], start_dim=0))
        a.plot(torch.flatten(x2[0], start_dim=0))
        plt.show()

        print(m(batch))

        i+=1
        if i == 10: break
