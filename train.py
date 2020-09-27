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
import util
import json
import train_model
import argparse

parser = argparse.ArgumentParser(
    description = 'train.py'
)

parser.add_argument('data_dir' , nargs = '?' , action = "store" , default = "./flowers/")
parser.add_argument('--pth' , dest = "pth" , action = "store" , default = "./checkpoint.pth")
parser.add_argument('--arch' , dest = "arch" , action = "store" , default = "densenet121" , type = str)
parser.add_argument('--epochs' , dest = "epochs" , action = "store" , default = 3 , type = int)
parser.add_argument('--lr' , dest = "lr" , action = "store" , default = 0.01 , type = float)
parser.add_argument('--hidden_layer' , dest = "hidden_layer" , action = "store" , default = 512 , type = int)
parser.add_argument('--device' , dest = "device" , action = "store" , default = "gpu" , type = str)
args = parser.parse_args()

train_datasets , valid_datasets , test_datasets , trainloaders , validloaders , testloaders = util.load_trans_data(args.data_dir)

model , optimizer , criterion = train_model.build_network(args.arch , args.hidden_layer , args.lr , args.device)

train_model.go_training(trainloaders , validloaders , model , optimizer , criterion , args.epochs , args.device)

train_model.saving_checkpoint(trainloaders , args.arch , args.hidden_layer)