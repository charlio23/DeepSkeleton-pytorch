import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from scipy.io import loadmat
import os
from PIL import Image
from scipy.ndimage.morphology import distance_transform_edt as bwdist
import numpy as np
import pandas as pd

class SKLARGE(Dataset):
    def __init__(self, rootDir, pathList):
        self.rootDir = rootDir
        self.listData =  pd.read_csv(pathList, dtype=str, delimiter=' ')

    def __len__(self):
        return len(self.listData)
                
    def __getitem__(self, i):
        # input and target images
        inputName = self.listData.iloc[i, 0]
        targetName = self.listData.iloc[i, 1]
        # process the images
        transf = transforms.ToTensor()
        inputImage = transf(Image.open(self.rootDir + inputName).convert('RGB'))
        """ CODE FOR RAW .MAT FILES
        itemGround = loadmat(self.rootDir + targetName)
        edge, skeleton = itemGround['edge'], itemGround['symmetry']
        dist = bwdist(1.0 - edge.astype(float))
        make_scale = np.vectorize(lambda x, y: 0 if y < 0.5 else x)
        #These should be parameters of the class
        receptive_fields = np.array([14,40,92,196])
        p = 1.2
        ####
        quantization = np.vectorize(lambda s: 0 if s < 0.001 else np.argmax(receptive_fields > p*s) + 1)
        scale = make_scale(dist,skeleton)
        quantise = quantization(scale)
        scaleTarget = torch.from_numpy(scale).float()
        quantiseTarget = torch.from_numpy(quantise)
        """
        targetImage = transf(Image.open(self.rootDir + targetName).convert('L'))*255.0
        return inputImage, targetImage

class SKLARGE_RAW(Dataset):
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
        inputImage = transf(Image.open(self.rootDir + inputName).convert('RGB'))
        itemGround = loadmat(self.rootDir + targetName)
        edge, skeleton = itemGround['edge'], itemGround['symmetry']
        dist = bwdist(1.0 - edge.astype(float))
        make_scale = np.vectorize(lambda x, y: 0 if y < 0.5 else x)
        scale = make_scale(dist,skeleton)
        scaleTarget = torch.from_numpy(scale).float()

        return inputImage, scaleTarget


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
        inputName = inputName.split(".jpg")[0] + ".png"
        return inputImage, inputName
