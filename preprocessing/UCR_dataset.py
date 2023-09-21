import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import math
import os

class UCRDataset(Dataset):
    def __init__(self,
                dataset_name: str,
                data_scaling: bool):
        
        #source: https://github.com/ML4ITS/TimeVQVAE/blob/main/preprocessing/preprocess_ucr.py
        
        self.root_dir = '../data/UCRArchive_2018/'
        
        assert os.path.exists(self.root_dir), 'UCRArchive_2018 does not exist.. please add it to the data folder'

        df_train = pd.read_csv(self.root_dir + f'{dataset_name}/' +f'/{dataset_name}_TRAIN.tsv', sep='\t', header=None)
        df_test = pd.read_csv(self.root_dir + f'{dataset_name}/' + f'/{dataset_name}_TEST.tsv', sep='\t', header=None)

        self.X_train, self.X_test = df_train.iloc[:, 1:].values, df_test.iloc[:, 1:].values
        self.Y_train, self.Y_test = df_train.iloc[:, [0]].values, df_test.iloc[:, [0]].values

        le = LabelEncoder()
        self.Y_train = le.fit_transform(self.Y_train.ravel())[:, None]
        self.Y_test = le.transform(self.Y_test.ravel())[:, None]

        if data_scaling:
            # following [https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries/blob/dcc674541a94ca8a54fbb5503bb75a297a5231cb/ucr.py#L30]
            mean = np.nanmean(self.X_train)
            var = np.nanvar(self.X_train)
            self.X_train = (self.X_train - mean) / math.sqrt(var)
            self.X_test = (self.X_test - mean) / math.sqrt(var)

        np.nan_to_num(self.X_train, copy=False)
        np.nan_to_num(self.X_test, copy=False)

        print('self.X_train.shape:', self.X_train.shape)
        print('self.X_test.shape:', self.X_test.shape)

        print("# unique labels (train):", np.unique(self.Y_train.reshape(-1)))
        print("# unique labels (test):", np.unique(self.Y_test.reshape(-1)))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import random

    data = UCRDataset('Wafer', data_scaling=True)

    x,y = data.X_train, data.Y_train

    n = random.randrange(0,len(x))
    plt.plot(np.arange(len(x[n])), x[n])
    plt.show()