import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from scipy.io import loadmat
import os
from PIL import Image
from scipy.ndimage.morphology import distance_transform_edt as bwdist
import numpy as np

class SKLARGE(Dataset):
    def __init__(self, rootDirImg, rootDirGt):
        self.rootDirImg = rootDirImg
        self.rootDirGt = rootDirGt
        self.listData = [sorted(os.listdir(rootDirImg)),sorted(os.listdir(rootDirGt))]

    def __len__(self):
        return len(self.listData[1])
                
    def __getitem__(self, i):
        # input and target images
        inputName = self.listData[0][i]
        targetName = self.listData[1][i]
        # process the images
        transf = transforms.ToTensor()
        inputImage = transf(Image.open(self.rootDirImg + inputName).convert('RGB'))
        itemGround = loadmat(self.rootDirGt + targetName)
        edge, skeleton = itemGround['edge'], itemGround['symmetry']
        dist = bwdist(1.0 - edge.astype(float))
        make_scale = np.vectorize(lambda x, y: 0 if y < 0.5 else x)
        #These should be parameters of the class
        receptive_fields = np.array([14,40,92,196])
        p = 1.2
        ####
        quantise = np.vectorize(lambda s: 0 if s < 0.001 else np.argmax(receptive_fields > p*s) + 1)
        scale = make_scale(dist,skeleton)
        quantization = quantise(scale)
        scaleTarget = torch.from_numpy(scale).float().unsqueeze_(0).int()
        quantiseTarget = torch.from_numpy(quantization).unsqueeze_(0).int()
        return inputImage, scaleTarget, quantiseTarget

"""
class SKLARGE_TEST(Dataset):
    def __init__(self, rootDirImg):
        self.rootDirImg = rootDirImg
        self.listData = sorted(os.listdir(rootDirImg))

    def __len__(self):
        return len(self.listData)
                
    def __getitem__(self, i):
        # input and target images
        inputName = self.listData[i]
        # process the images
        transf = transforms.ToTensor()
        inputImage = transf(Image.open(self.rootDirImg + inputName).convert('RGB'))
        inputName = inputName.split(".jpg")[0] + ".bmp"
        return inputImage, inputName
"""