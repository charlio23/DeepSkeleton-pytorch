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

demoPath = "data/images/test/42b43118f781e12791ed3021eb19d357.jpg"
testOutput = "images-demo/"

transf = transforms.ToTensor()
image = transf(Image.open(demoPath).convert('RGB')).unsqueeze_(0)

os.makedirs(testOutput, exist_ok=True)

print("Loading trained network...")

networkPath = "LMSDS.pth"

nnet = LMSDS().cuda()
dic = torch.load(networkPath)
dicli = list(dic.keys())
new = {}
j = 0

for k in nnet.state_dict():
    new[k] = dic[dicli[j]]
    j += 1

nnet.load_state_dict(new)

print("Generating test results...")
soft = torch.nn.Softmax(dim=1)
image = Variable(image).cuda()
sideOuts = nnet(image)

fuse = (1 - (soft(sideOuts[4])[0][0]).float()).unsqueeze_(0)
#fuse = grayTrans(fuse)
#plt.imshow(fuse)
#plt.show()
scale_outs = sideOuts[5:]
value, ind = soft(sideOuts[4]).max(1)
scale = torch.zeros(ind.size(1), ind.size(2))
receptive_fields = np.array([14,40,92,196])

for i in range(0,ind.size(1)):
    for j in range(0,ind.size(2)):
        index = ind[0,i,j] - 1
        if index < 0:
            scale[i,j] = 0
        else:
            scale[i,j] = (scale_outs[index][0,0,i,j] + 1)*receptive_fields[index]/2
print(scale.size())
img = Image.new("RGB",(len(scale[0]),len(scale)))
draw = ImageDraw.Draw(img)
for i in range(len(scale)):
    for j in range(len(scale[0])):
        r = scale[i][j]
        if scale[i][j] > 0.01:
            x = j
            y = i
            draw.ellipse((x-r, y-r, x+r, y+r), fill='blue')
plt.imshow(img)
plt.show(img)