import os
import torch
from dataset import SKLARGE_TEST
from model import LMSDS
from torch.autograd import Variable
from PIL import Image, ImageDraw
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms

def grayTrans(img):
    img = img.data.cpu().numpy()[0]*255.0
    img = (img).astype(np.uint8)
    img = Image.fromarray(img, 'L')
    return img

print("Loading train dataset...")
#0aa3fe72683525f40c580332172acb91
#42b43118f781e12791ed3021eb19d357
#0a83a27cddd7f016beee20585b90f1f8
demoPath = "SK-LARGE/aug_data/im_scale/1/o/0/f/0/0aa3fe72683525f40c580332172acb91.jpg"
gtPath = "SK-LARGE/aug_data/gt_scale/1/o/0/f/0/0aa3fe72683525f40c580332172acb91.png"
testOutput = "images-demo/"

transf = transforms.ToTensor()
image = transf(Image.open(demoPath).convert('RGB')).unsqueeze_(0)
scale = transf(Image.open(gtPath).convert('L'))*255.0

os.makedirs(testOutput, exist_ok=True)

print("Loading trained network...")

networkPath = "LMSDS.pth"

nnet = LMSDS()
dic = torch.load(networkPath)
dicli = list(dic.keys())
new = {}
j = 0

for k in nnet.state_dict():
    new[k] = dic[dicli[j]]
    j += 1

nnet.load_state_dict(new)

receptive_fields = np.array([14,40,92,196])
p = 1.2

def generate_quantise(quantise):
    result = []
    for i in range(1,5):
        result.append(quantise*(quantise <= i).long())

    result.append(quantise)
    return result

def generate_scales(quant_list, fields, scale):
    result = []
    for quant, r in zip(quant_list,fields):
        normalization = (2*((quant > 0).float()*scale)/r) - 1
        result.append(normalization)
        print("Scale: ", torch.max(quant))
        print("Scale for ", r, ", Max: ", torch.max(normalization), " Min: ", torch.min(normalization))

    return result

def apply_quantization(scale):
    if scale < 0.001:
        return 0
    if p*scale > np.max(receptive_fields):
        return len(receptive_fields)
    return np.argmax(receptive_fields > p*scale) + 1

quantization = np.vectorize(apply_quantization)
quantise = torch.from_numpy(quantization(scale.numpy())).squeeze_(1)
quant_list = generate_quantise(quantise)

scalee = Variable(scale)
scale_list = generate_scales(quant_list, receptive_fields, scalee)

print("Generating test results...")
soft = torch.nn.Softmax(dim=1)
image = Variable(image)
sideOuts = nnet(image)

fuse = (1 - (soft(sideOuts[4])[0][0]).float()).unsqueeze_(0)

scale_outs = sideOuts[5:]
value, ind = soft(sideOuts[4]).max(1)
scale = torch.zeros(ind.size(1), ind.size(2))
scale_fsds = torch.zeros(ind.size(1), ind.size(2))

for i in range(0,ind.size(1)):
    for j in range(0,ind.size(2)):
        index = quantise[0,i,j] - 1
        if index < 0:
            scale[i,j] = 0
        else:
            scale[i,j] = (scale_list[index][0,i,j] + 1)*receptive_fields[index]/2
img = Image.new("RGB",(len(scale[0]),len(scale)))
draw = ImageDraw.Draw(img)
scalee = scalee.squeeze_(0)
for i in range(len(scale)):
    for j in range(len(scale[0])):
        r = scale[i][j]/2
        if scale[i][j] > 0.01:
            x = j
            y = i
            draw.ellipse((x-r, y-r, x+r, y+r), fill='blue')

fig = plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(np.transpose(image[0].cpu().numpy(), (1, 2, 0)))
plt.show(fig)