
import collections
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time
import os
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
import time
t0 = time.time()

def get_imbalanced_data(dataset, num_sample_per_class, shuffle=False, random_seed=0):
    
    """
    Return a list of imbalanced indices from a dataset.
    Input: A dataset (e.g., CIFAR-10), num_sample_per_class: list of integers
    Output: imbalanced_list
    """
    
    length = dataset.__len__()
    num_sample_per_class = list(num_sample_per_class)
    selected_list = []
    indices = list(range(0,length))

    for i in range(0, length):
        index = indices[i]
        _, label = dataset.__getitem__(index)
        if num_sample_per_class[label] > 0:
            selected_list.append(index)
            num_sample_per_class[label] -= 1

    return selected_list

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
batch_size = 100

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
num_sample_per_class =  [4500, 2000, 1000, 800, 600, 500, 400, 250, 150, 80]
train_in_idx = get_imbalanced_data(trainset, num_sample_per_class)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=SubsetRandomSampler(train_in_idx), num_workers=8)

num_sample_per_class1 = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_in_idx = get_imbalanced_data(testset, num_sample_per_class1)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, sampler=SubsetRandomSampler(test_in_idx), num_workers=8)
numberofclass = 10

dec_x = []
dec_y = []
ct = 0
for x,y in test_loader:
    if ct == 0:
        dec_x = x
        dec_y = y
        print(dec_y)
        ct=1
        continue
    
    temp = y
    dec_x = torch.cat((dec_x,x),axis=0)
    dec_y = torch.cat((dec_y,temp),axis=0)

torch.save(dec_x, 'images_val.pt')
torch.save(dec_y, 'labels_val.pt')
