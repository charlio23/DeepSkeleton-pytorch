from dataset import SKLARGE
from model import initialize_lmsds
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from train import train, evaluate
import numpy as np

model_save_name = "LMSDS_VAL_"

print("Importing datasets...")

rootDir = "../SK-LARGE-VAL/"
trainListPath = "../SK-LARGE-VAL/aug_data/train_pair.lst"

trainDS = SKLARGE(rootDir, trainListPath)
train_data = DataLoader(trainDS, shuffle=True, batch_size=1, num_workers=4)

print("Initializing network...")

modelPath = "model/vgg16.pth"

print("Defining hyperparameters...")

### HYPER-PARAMETERS
learningRates = [1e-7]
p = 1.2
L = np.logspace(-4, 3, 15)
###
for learningRate in learningRates:
    for lamb in L:
        try:
            nnet = torch.nn.DataParallel(initialize_lmsds(modelPath, False)).cuda()
            nnet = train(nnet, train_data, p, learningRate, 6, lamb)
            name = str(lamb).replace('.','_')
            torch.save(nnet.state_dict(), model_save_name + name + '.pth')
        except ValueError:
            raise ValueError
