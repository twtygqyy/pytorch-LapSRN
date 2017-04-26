import torch.utils.data as data
import torch
import numpy as np
import h5py

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path)
        self.data = hf.get("data")
        self.label_x2 = hf.get("label_x2")
        self.label_x4 = hf.get("label_x4")

    def __getitem__(self, index):            
        return torch.from_numpy(self.data[index,:,:,:]).float(), torch.from_numpy(self.label_x2[index,:,:,:]).float(), torch.from_numpy(self.label_x4[index,:,:,:]).float()
        
    def __len__(self):
        return self.data.shape[0]