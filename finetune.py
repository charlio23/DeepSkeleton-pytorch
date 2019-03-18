from dataset import SKLARGE, SKLARGE_RAW
from model import initialize_fsds
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from train import train, evaluate
import numpy as np

print("Importing datasets...")

rootDir = "SK-LARGE/"
trainListPath = "SK-LARGE/aug_data/train_pair.lst"

trainDS = SKLARGE(rootDir, trainListPath)
train_data = DataLoader(trainDS, shuffle=True, batch_size=1, num_workers=4)

evalDS = SKLARGE_RAW("SK-LARGE/images/val", "SK-LARGE/groundTruth/val")
eval_data = DataLoader(evalDS, shuffle=False, batch_size=1, num_workers=4)

print("Initializing network...")

modelPath = "model/vgg16.pth"
nnet = torch.nn.DataParallel(initialize_fsds(modelPath)).cuda()

print("Defining hyperparameters...")

### HYPER-PARAMETERS
learningRates = [1e-6, 5e-7, 1e-7, 5e-8, 1e-8, 5e-9, 1e-9]
ps = np.linspace(1.1,1.8,10)
###
first = True
bestp = 0
bestlr = 0
minloss = 0
results = []
for learningRate in learningRates:
    for p in ps:
        nnet = train(nnet, train_data, p, learningRate, 3)
    
        loss = evaluate(nnet, eval_data)
        results.append("lr: " + str(learningRate) + ", p: " + str(p) + "loss: " + str(loss))
        if first or loss < minloss:
            first = False
            bestp = p
            bestlr = learningRate
            minloss = loss

print(results)
print("Best learning rate is: ",bestlr)
print("Best p is: ", bestp)