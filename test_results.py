import os
import torch
from dataset import SKLARGE_TEST
from model import LMSDS, FSDS
from torch.autograd import Variable
from PIL import Image, ImageDraw
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

receptive_fields = np.array([14,40,92,196])
p = 1.2

def obtain_scale_map(loc_map, scale_map):
    batch, _, height, width = loc_map.size()
    value, ind = loc_map[0,1:].max(0)
    probability_map = 2*(1 - soft(loc_map)[:,0:1]) - 1
    scale_map = [(scale_map[i] + 1)*receptive_fields[i]/2 for i in range(0,len(scale_map))]
    scale_map = torch.cat(scale_map, 1)
    scale_map = (scale_map.gather(1,ind.unsqueeze_(0).unsqueeze_(0)))
    result = torch.cat([probability_map, scale_map],1)
    return result

rootDirImgTest = "SK-LARGE/images/test/"
rootNms = "output-sklarge-001-nms/"
testOutput = "output-sklarge-001/"

test = os.listdir(rootDirImgTest)

os.makedirs(testOutput, exist_ok=True)

print("Loading trained network...")

networkPath = "LMSDS-001-061105.pth"

nnet = LMSDS()
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
    image = Variable(image, requires_grad=False).unsqueeze_(0)
    sideOuts = nnet(image)
    """
    skeleton = grayTrans((1 - soft(sideOuts[4])[0][0]).unsqueeze_(0))
    skeleton.save(testOutput + inputName)
    """
    scale_map = obtain_scale_map(sideOuts[4], sideOuts[5:])
    skeleton_nms = transf(Image.open(rootNms + inputName).convert('L'))
    skeleton_nms[skeleton_nms < 0.2] = 0
    skeleton_nms[skeleton_nms >= 0.2] = 1
    skeleton_nms = (skeleton_nms*scale_map[0,1:]).data.numpy()[0]
    segment = Image.new("L",(skeleton_nms.shape[1],skeleton_nms.shape[0]))
    draw = ImageDraw.Draw(segment)
    indices = np.argwhere(skeleton_nms > 0.01)
    for (i, j) in indices:
        r = skeleton_nms[i,j]/2
        x = j
        y = i
        draw.ellipse((x-r, y-r, x+r, y+r), fill='white')
    segment.save(testOutput + inputName)
    