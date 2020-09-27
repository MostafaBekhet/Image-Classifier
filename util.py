# Imports here

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import torch
import json
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets , models , transforms
import torchvision.models as models
from PIL import Image

def load_trans_data(data_dir):
    
    #set_directories
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(225),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir , transform = train_transforms)

    valid_datasets = datasets.ImageFolder(valid_dir , transform = valid_transforms)

    test_datasets = datasets.ImageFolder(test_dir , transform = test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloaders = torch.utils.data.DataLoader(train_datasets , batch_size = 64 , shuffle = True)

    validloaders = torch.utils.data.DataLoader(valid_datasets , batch_size = 64 , shuffle = True)

    testloaders = torch.utils.data.DataLoader(test_datasets , batch_size = 64 , shuffle = True)
    
    return train_datasets , valid_datasets , test_datasets , trainloaders , validloaders , testloaders

