from dataset import SKLARGE
from model import initialize_fsds
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from PIL import Image
from torch import sigmoid
from torch.nn.functional import cross_entropy
from torch.autograd import Variable
import time
from itertools import chain
from tqdm import tqdm
from torch.optim import lr_scheduler
from collections import defaultdict
from train import train

def grayTrans(img):
    img = img.data.cpu().numpy()[0]*255.0
    img = (img).astype(np.uint8)
    img = Image.fromarray(img, 'L')
    return img

print("Importing datasets...")

rootDir = "SK-LARGE/"
trainListPath = "SK-LARGE/aug_data/train_pair.lst"

trainDS = SKLARGE(rootDir, trainListPath)
train_data = DataLoader(trainDS, shuffle=True, batch_size=1, num_workers=4)

print("Initializing network...")

modelPath = "model/vgg16.pth"
nnet = torch.nn.DataParallel(initialize_fsds(modelPath)).cuda()

print("Defining hyperparameters...")

### HYPER-PARAMETERS
learningRate = 1e-6
p = 1.2
###

nnet = train(nnet, train_data, p, learningRate, 2)
