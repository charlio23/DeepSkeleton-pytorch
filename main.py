from dataset import SKLARGE
from model import initialize_fsds, initialize_lmsds
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from PIL import Image
from torch import sigmoid
from torch.nn.functional import cross_entropy, mse_loss
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

rootDir = "SK-LARGE/"
trainListPath = "SK-LARGE/aug_data/train_pair.lst"

trainDS = SKLARGE(rootDir, trainListPath)
train = DataLoader(trainDS, shuffle=True, batch_size=1, num_workers=4)

print("Initializing network...")

modelPath = "model/vgg16.pth"
nnet = torch.nn.DataParallel(initialize_lmsds(modelPath)).cuda()

print("Defining hyperparameters...")

### HYPER-PARAMETERS
learningRate = 1e-6
momentum = 0.9
lossWeight = 1
initializationNestedFilters = 0
initializationFusionWeights = 1/5
weightDecay = 0.0002
receptive_fields = np.array([14,40,92,196])
p = 1.2
###

# Optimizer settings.
net_parameters_id = defaultdict(list)
for name, param in nnet.named_parameters():
    print(name)
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
    elif name in ['module.sideOutLoc1.weight', 'module.sideOutLoc2.weight',
                  'module.sideOutLoc3.weight', 'module.sideOutLoc4.weight', 'module.sideOutLoc5.weight', 'module.sideOutScale1.weight', 'module.sideOutScale2.weight',
                  'module.sideOutScale3.weight', 'module.sideOutScale4.weight', 'module.sideOutScale5.weight']:
        print('{:26} lr: 0.01 decay:1'.format(name)); net_parameters_id['score_dsn_1-5.weight'].append(param)
    elif name in ['module.sideOutLoc1.bias', 'module.sideOutLoc2.bias',
                  'module.sideOutLoc3.bias', 'module.sideOutLoc4.bias', 'module.sideOutLoc5.bias', 'module.sideOutScale1.bias', 'module.sideOutScale2.bias',
                  'module.sideOutScale3.bias', 'module.sideOutScale4.bias', 'module.sideOutScale5.bias']:
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
    {'params': net_parameters_id['score_final.weight']  , 'lr': learningRate*5, 'weight_decay': weightDecay},
    {'params': net_parameters_id['score_final.bias']    , 'lr': learningRate*0, 'weight_decay': 0.},
], lr=learningRate, momentum=momentum, weight_decay=weightDecay)

# Learning rate scheduler.
lr_schd = lr_scheduler.StepLR(optimizer, step_size=1e5, gamma=0.1)

def balanced_cross_entropy(input, target):
    #weights
    weights = []
    for i in range(0,input.size(1)):
        w = torch.sum(target == i).item()
        if w < 1e-7:
            weights.append(0)
        else:
            weights.append(1.0/w)
    weight_total = sum(weights)
    weights = (torch.tensor(weights).float()/weight_total).cuda()
    #CE loss
    loss = cross_entropy(input,target,weight=weights,reduction='none')
    batch = target.shape[0]

    return torch.sum(loss)/batch

def regressor_loss(input, targetQuant, targetScale):
    #add weight and we are done
    loss = mse_loss(input,targetScale)

def generate_quantise(quantise):
    result = []
    for i in range(1,5):
        result.append(quantise*(quantise <= i).long())

    result.append(quantise)
    return result

def generate_scales(quant_list, fields, scale):
    result = []
    for quantise, r in zip(quant_list,fields):
        result.append(2*((quantise).float()*scale)/r - 1)

    return result

print("Training started")

epochs = 40
i = 0
dispInterval = 500
lossAcc = 0.0
train_size = 10
epoch_line = []
loss_line = []
nnet.train()
optimizer.zero_grad()
soft = torch.nn.Softmax(dim=1)

for epoch in range(epochs):
    print("Epoch: " + str(epoch + 1))
    for j, data in enumerate(tqdm(train), 1):
        image, scale = data
        image, scale = Variable(image).cuda(), Variable(scale).cuda()
        quantization = np.vectorize(lambda s: 0 if s < 0.001 else np.argmax(receptive_fields > p*s) + 1)
        quantise = torch.from_numpy(quantization(scale.cpu().numpy())).squeeze_(1).cuda()
        scale = scale.unsqueeze_(1)

        quant_list = generate_quantise(quantise)
        scale_list = generate_scales(quant_list, receptive_fields, scale)
        #scale = Variable(scale).cuda()
        sideOuts = nnet(image)
        
        loss = sum([balanced_cross_entropy(sideOut, quant) for sideOut, quant in zip(sideOuts,quant_list)])
        
        if np.isnan(float(loss.item())):
            raise ValueError('loss is nan while training')

        loss.backward()
        #lossAvg = loss/train_size
        #lossAvg.backward()
        lossAcc += loss.item()

        #if j % train_size == 0:
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

    for k in range(0,5):
        grayTrans((1 - soft(sideOuts[k])[0][0]).unsqueeze_(0)).save('images/sample_' + str(k+1) + '.png')
    grayTrans((quantise > 0.5)).save('images/sample_T.png')

    torch.save(nnet.state_dict(), 'FSDS.pth')
    plt.clf()
    plt.plot(epoch_line,loss_line)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("images/loss.png")
    plt.clf()