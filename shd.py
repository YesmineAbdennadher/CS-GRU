import tables
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

def generate_dataset(file_name,output_dir,dt=1e-3):
    fileh = tables.open_file(file_name, mode='r')
    units = fileh.root.spikes.units
    times = fileh.root.spikes.times
    labels = fileh.root.labels
    os.mkdir(output_dir)
    # This is how we access spikes and labels
    index = 0
    print("Number of samples: ",len(times))
    for i in range(len(times)):
        x_tmp = binary_image_readout(times[i], units[i],dt=dt)
        y_tmp = labels[i]
        output_file_name = os.path.join(output_dir, 'ID_' + str(i) + '_' + str(y_tmp) + '.npy')
        np.save(output_file_name, x_tmp)
    print('done..')
    return 0

def binary_image_readout(times,units,dt = 1e-3):
    img = []
    N = int(1/dt)
    for i in range(N):
        idxs = np.argwhere(times<=i*dt).flatten()
        vals = units[idxs]
        vals = vals[vals > 0]
        vector = np.zeros(700)
        vector[700-vals] = 1
        times = np.delete(times,idxs)
        units = np.delete(units,idxs)
        img.append(vector)
    return np.array(img)

class my_Dataset(Dataset):
    # Characterizes a dataset for PyTorch
    def __init__(self, data_paths, transform=None):
        self.data_paths = data_paths
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):   
        x = torch.from_numpy(np.load(self.data_paths[index]))
        y_ = self.data_paths[index].split('_')[-1]
        y_ = int(y_.split('.')[0])
        y_tmp= np.array([int(y_)])
        y = torch.from_numpy(y_tmp)
        if self.transform:
            x = self.transform(x)
        return x, y
