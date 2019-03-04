from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
import math
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat
from scipy.ndimage.morphology import distance_transform_edt as bwdist

rootDirGt = "data/groundTruth/train/"
rootDirImg = "data/images/train/"

ouputDir = "data/skeleton/train/"
os.makedirs(ouputDir, exist_ok=True)
listImages = sorted(os.listdir(rootDirImg))
listData = sorted(os.listdir(rootDirGt))

for i, itemName in enumerate(listData, 0):
    itemGround = loadmat(rootDirGt + itemName)
    edge, skeleton = itemGround['edge'], itemGround['symmetry']
    dist = bwdist(1 - edge)
    appl = np.vectorize(lambda x, y: y if x > 0.5 else 0)
    result = appl(skeleton,dist)
    plt.imshow(result)
    plt.show()


    savePath = listImages[i].split(".jpg")[0] + ".bmp"
    img = Image.fromarray(result.astype(np.uint8), 'L')
    img.save(ouputDir + savePath)

    print(np.max(Image.open(ouputDir + savePath)))
    break


