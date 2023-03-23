import os
import sys
# Import modules from base directory
sys.path.insert(0, os.getcwd())

import json
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as tsfm
from torch.utils.data import Dataset


def show_images(datset, num_samples=20, cols=4):
    """ Plots some samples from the dataset """
    plt.figure(figsize=(15,15)) 
    for i, img in enumerate(datset):
        if i == num_samples:
            break
        plt.subplot(num_samples/cols + 1, cols, i + 1)
        plt.imshow(img[0])


class Cars(Dataset):
    """
    Dataset of car images
    """
    def __init__(self, data_dir="", transform=None):
        super(Cars, self).__init__()

        self.transform = transform
        self.data = torchvision.datasets.StanfordCars(root=data_dir)


    def __getitem__(self, index):
        sample = tsfm.ToTensor()(self.data[index][0])
        sample *= 2
        sample -= 1

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.feature_list)


    def collate_fn(self, batch):
        imgs = list(zip(*batch))
        imgs = torch.stack(imgs, dim=0)

        return imgs


class CarsTest(Dataset):
    """
    Dataset of car images
    """
    def __init__(self, data_dir="", transform=None):
        super(Cars, self).__init__()

        self.transform = transform
        self.data = torchvision.datasets.StanfordCars(root=data_dir, split="test")


    def __getitem__(self, index):
        sample = tsfm.ToTensor()(self.data[index][0])
        sample *= 2
        sample -= 1

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.feature_list)


    def collate_fn(self, batch):
        imgs = list(zip(*batch))
        imgs = torch.stack(imgs, dim=0)

        return imgs
    

if __name__ == "main":
    print('hi')
    ds = CarsTest(data_dir=r"C:\Users\matth\OneDrive\Documents\Storage\Projects\DiffusionModel\data")
    print('bye')

