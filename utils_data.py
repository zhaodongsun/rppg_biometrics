import numpy as np
import os
import h5py
from torch.utils.data import Dataset
from scipy.fft import fft
from scipy import signal
from scipy.signal import butter, filtfilt
import glob
from numpy.random import default_rng

    
class H5Dataset(Dataset):
    # this dataset is used in the 1st training stage, only returning ST maps.
    def __init__(self, train_list, T):
        self.train_list = np.random.permutation(train_list) # list of .h5 file paths for training
        self.T = T # video clip length

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, idx):
        
        with h5py.File(self.train_list[idx], 'r') as f:
            img_length = int(f['imgs'].shape[0]) # first 60% for training

            idx_start = np.random.choice(img_length-self.T)
            idx_end = idx_start+self.T

            img_seq = f['imgs'][idx_start:idx_end]
            img_seq = np.transpose(img_seq, (3, 0, 1, 2)).astype('float32')
            # permutation
            hw = img_seq.shape[2]
            img_seq = img_seq.reshape(3, self.T, -1) 
            img_seq = img_seq[:,:,np.random.permutation(hw*hw)] # shape (3, T, S)
            img_seq = np.transpose(img_seq, (0,2,1)) # shape (3, S, T), ST map
        return img_seq


class H5Dataset_id(Dataset):
    # this dataset is used in the 2nd training stage, returning ST maps and ID labels.
    def __init__(self, train_list, T):
        self.train_list = train_list # list of .h5 file paths for training
        self.T = T # video clip length

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, idx):
        f_name = self.train_list[idx].split('/')[-1][:-3]
        id_label = int(f_name[:3])-1

        with h5py.File(self.train_list[idx], 'r') as f:
            img_length = int(np.min([f['imgs'].shape[0], f['bvp'].shape[0]])*0.6) # first 60% for training

            idx_start = np.random.choice(img_length-self.T)
            idx_end = idx_start+self.T

            img_seq = f['imgs'][idx_start:idx_end]
            img_seq = np.transpose(img_seq, (3, 0, 1, 2)).astype('float32')
            # permutation
            hw = img_seq.shape[2]
            img_seq = img_seq.reshape(3, self.T, -1) 
            img_seq = img_seq[:,:,np.random.permutation(hw*hw)] # shape (3, T, S)
            img_seq = np.transpose(img_seq, (0,2,1)) # shape (3, S, T), ST map
        return img_seq, id_label