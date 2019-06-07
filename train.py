from tqdm import tqdm
import torch
import numpy as np
from torch.optim import lr_scheduler
from collections import defaultdict
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.functional import cross_entropy

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
    batch, _, height, width = target.shape
    total_weight = height*width
    pos_weight = torch.sum(target > 0.1).item()/total_weight
    neg_weight = 1 - pos_weight
    weights = torch.ones(input.size(1))*neg_weight
    weights[0] = pos_weight
    #CE loss
    loss = cross_entropy(input,target.squeeze_(1),weight=weights.cuda(),reduction='none')

    return torch.sum(loss)/batch

def regressor_loss(input, targetScale, targetQuant):
    weight = (targetQuant > 0.01).float()
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

def train(nnet, train_data, p=1.2, learningRate=1e-6, nEpochs = 3, L = 0.5):
    momentum = 0.9
    weightDecay = 2e-4
    receptive_fields = np.array([14,40,92,196])

    def apply_quantization(scale):
        if scale < 0.001:
            return 0
        if p*scale > np.max(receptive_fields):
            return len(receptive_fields)
        return np.argmax(receptive_fields > p*scale) + 1

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

    print("Training started")
    train_size = 2
    nnet.train()
    soft = torch.nn.Softmax(dim=1)

    for epoch in range(nEpochs):
        print("Epoch: " + str(epoch + 1) + "   " + str(L))
        for j, data in enumerate(tqdm(train_data), 1):
            image, scale = data
            image = Variable(image).cuda()
            
            quantization = np.vectorize(apply_quantization)
            quantise = torch.from_numpy(quantization(scale.numpy())).cuda()
            quant_list = generate_quantise(quantise)
            
            scale = Variable(scale).cuda()
            scale_list = generate_scales(quant_list, receptive_fields, scale)

            sideOuts = nnet(image)
            quantise_SO = sideOuts[0:5]
            scale_SO = sideOuts[5:]
            loss_list = sum([balanced_cross_entropy(sideOut, quant) for sideOut, quant in zip(sideOuts,quant_list)])
            loss_list_scale = sum([regressor_loss(sideOut, scale, quant) for sideOut, scale, quant in zip(scale_SO,scale_list,quant_list[0:4])])
            loss = loss_list + L*sum(loss_list_scale)
            if np.isnan(float(loss.item())):
                raise ValueError('loss is nan while training')
            lossAvg = loss/train_size
            lossAvg.backward()
            if j % train_size == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_schd.step()

    return nnet