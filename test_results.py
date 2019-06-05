import os
import torch
from dataset import SKLARGE_TEST
from model import LMSDS, FSDS
from torch.autograd import Variable
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms

def grayTrans(img):
    img = img.data.cpu().numpy()[0]*255.0
    img = (img).astype(np.uint8)
    img = Image.fromarray(img, 'L')
    return img

print("Loading train dataset...")

rootDirImgTest = "SK-LARGE/images/test/"
testOutput = "output/"

test = os.listdir(rootDirImgTest)

os.makedirs(testOutput, exist_ok=True)

print("Loading trained network...")

networkPath = "LMSDS-0005-06138.pth"

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

transf = transforms.ToTensor()

for inputName in tqdm(test):
    inputImage = transf(Image.open(rootDirImgTest + inputName).convert('RGB'))
    inputName = inputName.replace(".jpg", ".png")
    tensorBlue = (inputImage[0:1, :, :] * 255.0) - 104.00698793
    tensorGreen = (inputImage[1:2, :, :] * 255.0) - 116.66876762
    tensorRed = (inputImage[2:3, :, :] * 255.0) - 122.67891434
    image = torch.cat([ tensorBlue, tensorGreen, tensorRed ], 0)
    image = Variable(image, requires_grad=False).cuda().unsqueeze_(0)
    sideOuts = nnet(image)
    fuse = grayTrans((1 - soft(sideOuts[4])[0][0]).unsqueeze_(0))
    fuse.save(testOutput + inputName)