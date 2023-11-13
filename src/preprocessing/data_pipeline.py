from torch.utils.data import DataLoader
from src.preprocessing.preprocess_ucr import UCRDataset, AugUCRDataset, UCRDatasetImporter
from src.preprocessing.augmentations import Augmentations


def build_data_pipeline(batch_size, dataset_importer: UCRDatasetImporter, config: dict, kind: str, augmentations=[], n_pairs=2, shuffle_train=True) -> DataLoader:
    """
    :param config:
    :param kind train/valid/test
    """
    num_workers = config['dataset']["num_workers"]

    if len(augmentations) == 0:
        # DataLoader
        if kind == 'train':
            train_dataset = UCRDataset("train", dataset_importer)
            return DataLoader(train_dataset, batch_size, num_workers=num_workers, shuffle=shuffle_train, drop_last=False, pin_memory=True)  # `drop_last=False` due to some datasets with a very small dataset size.
        elif kind == 'test':
            test_dataset = UCRDataset("test", dataset_importer)
            return DataLoader(test_dataset, batch_size, num_workers=num_workers, shuffle=False, drop_last=False, pin_memory=True)
        else:
            raise ValueError
    else:
        augs = Augmentations()
        # DataLoader
        if kind == 'train':
            train_dataset = AugUCRDataset("train", dataset_importer, augs, augmentations, n_pairs=n_pairs)
            return DataLoader(train_dataset, batch_size, num_workers=num_workers, shuffle=shuffle_train, drop_last=False, pin_memory=True)  # `drop_last=False` due to some datasets with a very small dataset size.
        elif kind == 'test':
            test_dataset = AugUCRDataset("test", dataset_importer, augs, [], subseq_lens=n_pairs)
            return DataLoader(test_dataset, batch_size, num_workers=num_workers, shuffle=False, drop_last=False, pin_memory=True)
        else:
            raise ValueError