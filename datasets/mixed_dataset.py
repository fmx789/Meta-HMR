"""
This file contains the definition of different heterogeneous datasets used for training
"""
import torch
import numpy as np

from .base_dataset import BaseDataset

class MixedDataset(torch.utils.data.Dataset):

    def __init__(self, options, **kwargs):
        self.dataset_list = ['h36m', 'coco', 'mpii', 'mpi-inf-3dhp', '3dpw']
        self.dataset_dict = {'h36m': 0, 'coco': 1, 'mpii': 2, 'mpi-inf-3dhp': 3, '3dpw': 4}
        self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
        print([len(ds) for ds in self.datasets])
        total_length = sum([len(ds) for ds in self.datasets])
        length_itw = sum([len(ds) for ds in self.datasets[1:-2]])
        self.length = max([len(ds) for ds in self.datasets])
        """
        Data distribution inside each batch:
        40% H36M - 30% ITW - 10% MPI-INF - 20% 3DPW
        """
        self.partition = [.4, .3*len(self.datasets[1])/length_itw,
                          .3*len(self.datasets[2])/length_itw,
                          0.1,
                          0.2,
                          ]
        self.partition = np.array(self.partition).cumsum()

    def __getitem__(self, index):
        p = np.random.rand()
        for i in range(len(self.dataset_list)):
            if p <= self.partition[i]:
                return self.datasets[i][index % len(self.datasets[i])]

    def __len__(self):
        return self.length
    

class MixedDataset_wo3dpw(torch.utils.data.Dataset):

    def __init__(self, options, **kwargs):
        self.dataset_list = ['h36m', 'coco', 'mpii', 'mpi-inf-3dhp']
        self.dataset_dict = {'h36m': 0, 'coco': 1, 'mpii': 2, 'mpi-inf-3dhp': 3}
        self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
        print([len(ds) for ds in self.datasets])
        total_length = sum([len(ds) for ds in self.datasets])
        length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
        print('itw length:', length_itw)
        self.length = max([len(ds) for ds in self.datasets])
        """
        Data distribution inside each batch:
        50% H36M - 30% ITW - 20% MPI-INF
        """
        self.partition = [.5, .3*len(self.datasets[1])/length_itw,
                          .3*len(self.datasets[2])/length_itw,
                          0.2,
                          ]
        self.partition = np.array(self.partition).cumsum()

    def __getitem__(self, index):
        p = np.random.rand()
        for i in range(len(self.dataset_list)):
            if p <= self.partition[i]:
                return self.datasets[i][index % len(self.datasets[i])]

    def __len__(self):
        return self.length
