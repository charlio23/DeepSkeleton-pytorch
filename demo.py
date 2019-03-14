import os
import torch
from dataset import SKLARGE_TEST
from model import LMSDS
from torch.autograd import Variable
from PIL import Image
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

fuse = grayTrans(((soft(sideOuts[4])[0][4])).unsqueeze_(0))
plt.imshow(fuse)
plt.show()