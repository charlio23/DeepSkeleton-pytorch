import os
import torch
from dataset import SKLARGE_TEST
from model import LMSDS, FSDS
from torch.autograd import Variable
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

def grayTrans(img):
    img = img.data.cpu().numpy()[0]*255.0
    img = (img).astype(np.uint8)
    img = Image.fromarray(img, 'L')
    return img

print("Loading train dataset...")

rootDirImgTest = "SK-LARGE/images/test/"
testOutput = "output/"

testDS = SKLARGE_TEST(rootDirImgTest)
test = DataLoader(testDS, shuffle=False)

os.makedirs(testOutput, exist_ok=True)

print("Loading trained network...")

networkPath = "FSDS.pth"

nnet = FSDS()
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

for data in tqdm(test):
    image, imgName = data
    image = Variable(image, requires_grad=False)
    sideOuts = nnet(image)
    fuse = grayTrans((1 - soft(sideOuts[4])[0][0]).unsqueeze_(0))
    fuse.save(testOutput + imgName[0])