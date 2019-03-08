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

def grayTrans(img):
    img = img.data.cpu().numpy()[0]*255.0
    img = (img).astype(np.uint8)
    img = Image.fromarray(img, 'L')
    return img

print("Importing datasets...")

rootDirImgTrain = "data/images/train/"
rootDirGtTrain = "data/groundTruth/train/"
rootDirImgTest = "data/images/test/"
rootDirGtTest = "data/groundTruth/test/"

trainDS = SKLARGE(rootDirImgTrain, rootDirGtTrain)
#trainDS = ConcatDataset([trainDS,valDS])

print("Initializing network...")


modelPath = "model/vgg16.pth"


nnet = torch.nn.DataParallel(initialize_fsds(modelPath)).cuda()

train = DataLoader(trainDS, shuffle=True, batch_size=1, num_workers=4)


print("Defining hyperparameters...")

### HYPER-PARAMETERS
learningRate = 1e-6
momentum = 0.9
lossWeight = 1
initializationNestedFilters = 0
initializationFusionWeights = 1/5
weightDecay = 0.0002
###

# Optimizer settings.
net_parameters_id = defaultdict(list)
for name, param in nnet.named_parameters():
    if name in ['module.conv1.0.weight', 'module.conv1.2.weight',
                'module.conv2.1.weight', 'module.conv2.3.weight',
                'module.conv3.1.weight', 'module.conv3.3.weight', 'module.conv3.5.weight',
                'module.conv4.1.weight', 'module.conv4.3.weight', 'module.conv4.5.weight']:
        print('{:26} lr:    1 decay:1'.format(name)); net_parameters_id['conv1-4.weight'].append(param)
    elif name in ['module.conv1.0.bias', 'module.conv1.2.bias',
                'module.conv2.1.bias', 'module.conv2.3.bias',
                'module.conv3.1.bias', 'module.conv3.3.bias', 'module.conv3.5.bias',
                'module.conv4.1.bias', 'module.conv4.3.bias', 'module.conv4.5.bias']:
        print('{:26} lr:    2 decay:0'.format(name)); net_parameters_id['conv1-4.bias'].append(param)
    elif name in ['module.conv5.1.weight', 'module.conv5.3.weight', 'module.conv5.5.weight']:
        print('{:26} lr:  100 decay:1'.format(name)); net_parameters_id['conv5.weight'].append(param)
    elif name in ['module.conv5.1.bias', 'module.conv5.3.bias', 'module.conv5.5.bias']:
        print('{:26} lr:  200 decay:0'.format(name)); net_parameters_id['conv5.bias'].append(param)
    elif name in ['module.sideOut1.weight', 'module.sideOut2.weight',
                  'module.sideOut3.weight', 'module.sideOut4.weight', 'module.sideOut5.weight']:
        print('{:26} lr: 0.01 decay:1'.format(name)); net_parameters_id['score_dsn_1-5.weight'].append(param)
    elif name in ['module.sideOut1.bias', 'module.sideOut2.bias',
                  'module.sideOut3.bias', 'module.sideOut4.bias', 'module.sideOut5.bias']:
        print('{:26} lr: 0.02 decay:0'.format(name)); net_parameters_id['score_dsn_1-5.bias'].append(param)
    elif name in ['module.fuseScale0.weight', 'module.fuseScale1.weight',
                  'module.fuseScale2.weight', 'module.fuseScale3.weight']:
        print('{:26} lr:0.001 decay:1'.format(name)); net_parameters_id['score_final.weight'].append(param)
    elif name in ['module.fuseScale0.bias', 'module.fuseScale1.bias',
                  'module.fuseScale2.bias', 'module.fuseScale3.bias']:
        print('{:26} lr:0.002 decay:0'.format(name)); net_parameters_id['score_final.bias'].append(param)

# IMPORTANT: In the official implementation paper, they specify that the lr is 5 times the base learning rate, contrary to their caffe code version

# Create optimizer.
optimizer = torch.optim.SGD([
    {'params': net_parameters_id['conv1-4.weight']      , 'lr': learningRate*1    , 'weight_decay': weightDecay},
    {'params': net_parameters_id['conv1-4.bias']        , 'lr': learningRate*2    , 'weight_decay': 0.},
    {'params': net_parameters_id['conv5.weight']        , 'lr': learningRate*100  , 'weight_decay': weightDecay},
    {'params': net_parameters_id['conv5.bias']          , 'lr': learningRate*200  , 'weight_decay': 0.},
    {'params': net_parameters_id['score_dsn_1-5.weight'], 'lr': learningRate*0.01 , 'weight_decay': weightDecay},
    {'params': net_parameters_id['score_dsn_1-5.bias']  , 'lr': learningRate*0.02 , 'weight_decay': 0.},
    {'params': net_parameters_id['score_final.weight']  , 'lr': learningRate*0.001, 'weight_decay': weightDecay},
    {'params': net_parameters_id['score_final.bias']    , 'lr': learningRate*0.002, 'weight_decay': 0.},
], lr=learningRate, momentum=momentum, weight_decay=weightDecay)

# Learning rate scheduler.
lr_schd = lr_scheduler.StepLR(optimizer, step_size=1e4, gamma=0.1)

def balanced_cross_entropy(input, target):
    #weights
    weights = []
    for i in range(0,input.size(1)):
        weights.append(torch.sum(target == i).item())
    weight_total = sum(weights)
    weights = (torch.tensor(weights).float()/weight_total).cuda()
    #CE loss
    loss = cross_entropy(input,target,weight=weights,reduction='none')
    batch = target.shape[0]

    return torch.sum(loss)/batch

def generate_quantise(quantise):
    result = []
    for i in range(1,5):
        print((quantise*(quantise <= i).long()).max())
        result.append(quantise*(quantise <= i).long())

    result.append(quantise)
    return result

print("Training started")

epochs = 20
i = 0
dispInterval = 500
lossAcc = 0.0
train_size = 10
epoch_line = []
loss_line = []
nnet.train()
optimizer.zero_grad()

for epoch in range(epochs):
    print("Epoch: " + str(epoch + 1))
    for j, data in enumerate(tqdm(train), 0):
        image, scale, quantise = data
        image, scale, quantise = Variable(image).cuda(), Variable(scale).cuda(), Variable(quantise).cuda()
        sideOuts = nnet(image)
        quant_list = generate_quantise(quantise)
        loss = sum([balanced_cross_entropy(sideOut, quant) for sideOut, quant in zip(sideOuts,quant_list)])

        break
        
        lossAvg = loss/train_size
        lossAvg.backward()
        lossAcc += loss.item()

        if (j+1) % train_size == 0:
            optimizer.step()
            optimizer.zero_grad()
            lr_schd.step()
        if (i+1) % dispInterval == 0:
            timestr = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            lossDisp = lossAcc/dispInterval
            epoch_line.append(epoch + j/len(train))
            loss_line.append(lossDisp)
            print("%s epoch: %d iter:%d loss:%.6f"%(timestr, epoch+1, i+1, lossDisp))
            lossAcc = 0.0
        i += 1

    plt.imshow(np.transpose(image[0].cpu().numpy(), (1, 2, 0)))
    plt.savefig("images/sample_0.png")

    # transform to grayscale images

    for i in range(0,5):
        grayTrans((sideOuts[i].max(1)[1]) > 0.5).save('images/sample_' + str(i+1) + '.png')
    grayTrans((quantise > 0.5)).save('images/sample_T.png')

    torch.save(nnet.state_dict(), 'FSDS.pth')
    plt.clf()
    plt.plot(epoch_line,loss_line)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("images/loss.png")
    plt.clf()

    exit()