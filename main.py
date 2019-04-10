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
learningRate = 1e-7
momentum = 0.9
weightDecay = 0.0002
receptive_fields = np.array([14,40,92,196])
p = 1.2
L = 0.1
###

# Optimizer settings.
net_parameters_id = defaultdict(list)
for name, param in nnet.named_parameters():
    if name in ['module.conv1.0.weight', 'module.conv1.2.weight',
                'module.conv2.1.weight', 'module.conv2.3.weight',
                'module.conv3.1.weight', 'module.conv3.3.weight',
                'module.conv3.5.weight', 'module.conv4.1.weight',
                'module.conv4.3.weight', 'module.conv4.5.weight']:
        print('{:26} lr:    1 decay:1'.format(name)); net_parameters_id['conv1-4.weight'].append(param)
    elif name in ['module.conv1.0.bias', 'module.conv1.2.bias',
                  'module.conv2.1.bias', 'module.conv2.3.bias',
                  'module.conv3.1.bias', 'module.conv3.3.bias',
                  'module.conv3.5.bias', 'module.conv4.1.bias',
                  'module.conv4.3.bias', 'module.conv4.5.bias']:
        print('{:26} lr:    2 decay:0'.format(name)); net_parameters_id['conv1-4.bias'].append(param)
    elif name in ['module.conv5.1.weight', 'module.conv5.3.weight',
                  'module.conv5.5.weight']:
        print('{:26} lr:  100 decay:1'.format(name)); net_parameters_id['conv5.weight'].append(param)
    elif name in ['module.conv5.1.bias', 'module.conv5.3.bias',
                  'module.conv5.5.bias']:
        print('{:26} lr:  200 decay:0'.format(name)); net_parameters_id['conv5.bias'].append(param)
    elif name in ['module.sideOut1.weight', 'module.sideOut2.weight',
                  'module.fuseScale2.weight', 'module.sideOut3.weight',
                  'module.sideOut4.weight', 'module.sideOut5.weight',
                  'module.sideOutLoc1.weight', 'module.sideOutLoc2.weight',
                  'module.sideOutLoc3.weight','module.sideOutLoc4.weight',
                  'module.sideOutLoc5.weight','module.sideOutScale1.weight',
                  'module.sideOutScale2.weight','module.sideOutScale3.weight',
                  'module.sideOutScale4.weight','module.sideOutScale5.weight']:
        print('{:26} lr: 0.01 decay:1'.format(name)); net_parameters_id['score_dsn_1-5.weight'].append(param)
    elif name in ['module.sideOut1.bias', 'module.sideOut2.bias',
                  'module.sideOut3.bias', 'module.sideOut4.bias',
                  'module.sideOut5.bias','module.sideOutLoc1.bias', 
                  'module.sideOutLoc2.bias','module.sideOutLoc3.bias',
                  'module.sideOutLoc4.bias','module.sideOutLoc5.bias',
                  'module.sideOutScale1.bias','module.sideOutScale2.bias',
                  'module.sideOutScale3.bias', 'module.sideOutScale4.bias',
                  'module.sideOutScale5.bias']:
        print('{:26} lr: 0.02 decay:0'.format(name)); net_parameters_id['score_dsn_1-5.bias'].append(param)
    elif name in ['module.fuseScale0.weight', 'module.fuseScale1.weight',
                  'module.fuseScale3.weight','module.fuseScale3.weight']:
        print('{:26} lr:0.05 decay:1'.format(name)); net_parameters_id['score_final.weight'].append(param)
    elif name in ['module.fuseScale0.bias', 'module.fuseScale1.bias',
                  'module.fuseScale2.bias', 'module.fuseScale3.bias',
                  'module.fuseScale4.bias']:
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
    {'params': net_parameters_id['score_final.weight']  , 'lr': learningRate*0.05, 'weight_decay': weightDecay},
    {'params': net_parameters_id['score_final.bias']    , 'lr': learningRate*0.002, 'weight_decay': 0.},
], lr=learningRate, momentum=momentum, weight_decay=weightDecay)

# Learning rate scheduler.
lr_schd = lr_scheduler.StepLR(optimizer, step_size=3e4, gamma=0.1)

def balanced_cross_entropy(input, target):
    # weights original paper implementation
    """
    weights = []
    for i in range(0,input.size(1)):
        w = torch.sum(target == i).item()
        if w < 1e-7:
            weights.append(0)
        else:
            weights.append(1.0/w)
    weight_total = sum(weights)
    weights = (torch.tensor(weights).float()/weight_total).cuda()
    """
    # weights caffe code implementation
    batch, height, width = target.shape
    total_weight = batch*height*width
    pos_weight = torch.sum(target > 0.1).item()/total_weight
    neg_weight = 1 - pos_weight
    weights = torch.ones(input.size(1))*neg_weight
    weights[0] = pos_weight
    #CE loss
    loss = cross_entropy(input,target,weight=weights.cuda(),reduction='none')

    return torch.sum(loss)/batch

def regressor_loss(input, targetScale, targetQuant):
    weight = (targetQuant > 0.01).unsqueeze_(1).float()
    weight_total = torch.sum(weight)
    loss = torch.sum(weight*mse_loss(input, targetScale, reduction='none'))
    batch = targetScale.shape[0]
    return loss/batch


def generate_quantise(quantise):
    result = []
    for i in range(1,5):
        result.append(quantise*(quantise <= i).long())

    result.append(quantise)
    return result

def generate_scales(quant_list, fields, scale):
    result = []
    for quantise, r in zip(quant_list,fields):
        result.append(2*((quantise > 0).float()*scale)/r - 1)
    return result

def apply_quantization(scale):
    if scale < 0.001:
        return 0
    if p*scale > np.max(receptive_fields):
        return len(receptive_fields)
    return np.argmax(receptive_fields > p*scale) + 1

print("Training started")

epochs = 40
i = 0
dispInterval = 1000
lossAcc = [0.0]*10
train_size = 2
epoch_line = []
loss_line = [[], [], [], [], [], [], [], [], [], []]
nnet.train()
optimizer.zero_grad()
soft = torch.nn.Softmax(dim=1)
time_data = []
time_network = []
time_loss = []

for epoch in range(epochs):
    print("Epoch: " + str(epoch + 1))
    for j, data in enumerate(tqdm(train), 1):
        if j != 1:
            end = time.time()
            time_data.append(end - start)
        start = time.time()
        image, scale = data
        image = Variable(image).cuda()
        
        quantization = np.vectorize(apply_quantization)
        quantise = torch.from_numpy(quantization(scale.numpy())).squeeze_(1).cuda()
        quant_list = generate_quantise(quantise)
        
        scale = Variable(scale).cuda()
        scale_list = generate_scales(quant_list, receptive_fields, scale)

        sideOuts = nnet(image)
        quantise_SO = sideOuts[0:5]
        scale_SO = sideOuts[5:]
        end = time.time()
        time_network.append(end - start)
        start = time.time()
        loss_list = [balanced_cross_entropy(sideOut, quant) for sideOut, quant in zip(sideOuts,quant_list)]
        loss_list_scale = [regressor_loss(sideOut, scale, quant) for sideOut, scale, quant in zip(scale_SO,scale_list,quant_list[0:4])]

        loss = sum(loss_list) + L*sum(loss_list_scale)

        if np.isnan(float(loss.item())):
            raise ValueError('loss is nan while training')

        lossAvg = loss/train_size
        lossAvg.backward()
        
        end = time.time()
        time_loss.append(end - start)
        if j % train_size == 0:
            optimizer.step()
            optimizer.zero_grad()
            lr_schd.step()
            print("Loss time: ", np.average(time_loss))
            print("Network time: ", np.average(time_network))
            print("Data time: ", np.average(time_data))
        for l in range(0,5):
            lossAcc[l] += loss_list[l].clone().item()
        for l in range(5,9):
            lossAcc[l] += loss_list_scale[l - 5].clone().item()
        lossAcc[9] += loss.clone().item()

        if (i+1) % dispInterval == 0:

            timestr = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            epoch_line.append(epoch + j/len(train))
            for l in range(0,10):
                lossDisp = lossAcc[l]/dispInterval
                loss_line[l].append(lossDisp)
                lossAcc[l] = 0.0
                if l == 9:
                    print("%s epoch: %d iter:%d loss:%.6f"%(timestr, epoch+1, i+1, lossDisp))
        i += 1

    plt.imshow(np.transpose(image[0].cpu().numpy(), (1, 2, 0)))
    plt.savefig("images/sample_0.png")

    # transform to grayscale images

    for k in range(0,5):
        grayTrans((1 - soft(sideOuts[k])[0][0]).unsqueeze_(0)).save('images/sample_' + str(k+1) + '.png')
    grayTrans((quantise > 0.5)).save('images/sample_T.png')

    torch.save(nnet.state_dict(), 'LMSDS.pth')
    plt.clf()
    for l in range(0,9):
        if l == 4:
            continue
        plt.plot(epoch_line,loss_line[l])
        plt.xlabel("Epoch")
        plt.ylabel("Loss SO " + str(l+2))
        plt.savefig("images/loss_SO_" + str(l+2) + ".png")
        plt.clf()
    plt.plot(epoch_line,loss_line[4])
    plt.xlabel("Epoch")
    plt.ylabel("Loss Fuse")
    plt.savefig("images/loss_Fuse.png")
    plt.clf()
    plt.plot(epoch_line,loss_line[9])
    plt.xlabel("Epoch")
    plt.ylabel("Loss Total")
    plt.savefig("images/loss_Total.png")
    plt.clf()
