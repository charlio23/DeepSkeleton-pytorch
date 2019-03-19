from dataset import SKLARGE, SKLARGE_RAW
from model import initialize_fsds
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from train import train, evaluate
import numpy as np

print("Importing datasets...")

rootDir = "SK-LARGE/"
trainListPath = "train_pair_val.lst"

trainDS = SKLARGE(rootDir, trainListPath)
train_data = DataLoader(trainDS, shuffle=True, batch_size=1, num_workers=4)

evalDS = SKLARGE_RAW("SK-LARGE/images/val", "SK-LARGE/groundTruth/val")
eval_data = DataLoader(evalDS, shuffle=False, batch_size=1, num_workers=4)

print("Initializing network...")

modelPath = "model/vgg16.pth"

print("Defining hyperparameters...")

### HYPER-PARAMETERS
learningRates = [1e-7]
ps = np.linspace(1.2,1.8,10)
###
first = True
bestp = 0
bestlr = 0
minloss = 0
results = []
for learningRate in learningRates:
    for p in ps:
        try:
            nnet = torch.nn.DataParallel(initialize_fsds(modelPath)).cuda()
            nnet = train(nnet, train_data, p, learningRate, 3)
            loss = evaluate(nnet, eval_data)
            results.append("lr: " + str(learningRate) + ", p: " + str(p) + "loss: " + str(loss))
            if first or loss < minloss:
                first = False
                bestp = p
                bestlr = learningRate
                minloss = loss
        except:
            pass

print(results)
print("Best learning rate is: ",bestlr)
print("Best p is: ", bestp)