from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
import pandas as pd
from PIL import Image, ImageDraw
import os
import matplotlib.pyplot as plt
import math
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat
from scipy.ndimage.morphology import distance_transform_edt as bwdist

rootDirGt = "SK-LARGE/groundTruth/train/"
rootDirImg = "SK-LARGE/images/train/"

listData = sorted(os.listdir(rootDirGt))
listImages = sorted(os.listdir(rootDirImg))

index = np.random.randint(len(listData))
image = np.array(Image.open(rootDirImg + listImages[index]))
itemGround = loadmat(rootDirGt + listData[index])
edge, skeleton = itemGround['edge'], itemGround['symmetry']
dist = bwdist(1 - edge)
appl = np.vectorize(lambda x, y: 0 if y < 0.5 else x)
result = appl(dist,skeleton)

mask = skeleton > 0
image[mask] = np.array([0.0,255.0,255.0])

img = Image.new("RGB",(len(result[0]),len(result)))
draw = ImageDraw.Draw(img)
for i in range(len(result)):
	for j in range(len(result[0])):
		r = result[i][j]
		if result[i][j] > 0.01:
			x = j
			y = i
			draw.ellipse((x-r, y-r, x+r, y+r), fill='blue')
for i in range(len(result)):
	for j in range(len(result[0])):
		r = result[i][j]
		if result[i][j] > 0.01:
			draw.point((j,i),fill="yellow")
plt.imshow(img)
plt.show(img)


