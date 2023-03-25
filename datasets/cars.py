import os
import sys
# Import modules from base directory
sys.path.insert(0, os.getcwd())

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

        train = torchvision.datasets.StanfordCars(root=data_dir, transform=transform)
        test = torchvision.datasets.StanfordCars(root=data_dir, split="test", transform=transform)

        self.data = torch.utils.data.ConcatDataset([train, test])


    def __getitem__(self, index):
        return self.data[index][0] * 2.0 - 1.0

    def __len__(self):
        return len(self.data)


    def collate_fn(self, batch):
        return torch.stack(batch, dim=0)


class CarsTest(Dataset):
    """
    Dataset of car images
    """
    def __init__(self, data_dir="", transform=None):
        super(CarsTest, self).__init__()

        train = torchvision.datasets.StanfordCars(root=data_dir, transform=transform)
        test = torchvision.datasets.StanfordCars(root=data_dir, split="test", transform=transform)

        self.data = torch.utils.data.ConcatDataset([train, test])


    def __getitem__(self, index):
        return self.data[index][0] * 2.0 - 1.0

    def __len__(self):
        return len(self.data)


    def collate_fn(self, batch):
        return torch.stack(batch, dim=0)
    

# if __name__ == "main":
#     print('hi')
#     ds = CarsTest(data_dir=r"C:\Users\matth\OneDrive\Documents\Storage\Projects\DiffusionModel\data")
#     print('bye')

