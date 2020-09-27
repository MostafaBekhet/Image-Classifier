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

with open('car_to_name' , 'r') as f:
    cat_to_name = json.load(f)

parser = argparser.ArgumentParser(
    description = 'predict.py'
)

parser.add_argument('input' , nargs = '?' , action = "store" , default = "./flowers/test/1/image_06754.jpg" , type = str)
parser.add_argument('--pth' , dest = "pth" , action = "store" , default = "./checkpoint.pth")
parser.add_argument('--device' , dest = "device" , action = "store" , default = "gpu" , type = str)
parser.add_argument('--topk' , dest = "topk" , action = "store" , default = 3 , type = int)
parser.add_argument('cat_name' , dest = "cat_name" , action = "store" , default = 'cat_to_name.json')
args = parser.parse_args()

model = train_model.load_model(args.pth)


ps = train_model.predict(args.input , model , args.topk , args.device)

labels = [cat_to_name[str(index + 1)] for index in np.array(ps[1][0])]
ps = np.array(ps[0][0])

i = 0

while i < args.topk:
    print("{} with a probability od {}".format(labels[i] , ps[i]))
    i += 1
    
print("finaly finished")