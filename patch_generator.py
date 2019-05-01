import os
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np

def generatePatches(edgePath, skeletonPath, numPatches):

    edge = np.array(Image.open(edgePath))
    ske = np.array(Image.open(skeletonPath))

    composeImg = Image.fromarray(ske + edge)

    candidates = np.argwhere(ske > 0.1)
    num_candidates =  len(candidates)
    x_lim, y_lim = composeImg.size
    patches = []
    for _ in range(numPatches):
        candidate = candidates[np.random.randint(num_candidates)]

        x_coord, y_coord = candidate[0],candidate[1]
        width = np.ceil(2*ske[x_coord, y_coord])

        x_ini = int(np.floor(x_coord - width/2))
        x_end = int(np.ceil(x_coord + width/2))
        y_ini = int(np.floor(y_coord - width/2))
        y_end = int(np.ceil(y_coord + width/2))

        #check if coordinates are inside image and add padding
        if x_ini < 0 or y_ini < 0 or x_end > x_lim or y_end > y_lim:
            padding = np.max([-x_ini, -y_ini, x_end-x_lim, y_end-y_lim])
            x_ini += padding
            x_end += padding
            y_ini += padding
            y_end += padding
            paddingImg = ImageOps.expand(composeImg, padding)
            patchImg = Image.fromarray(np.array(paddingImg)[x_ini:x_end, y_ini:y_end])
        else:
            patchImg = Image.fromarray(np.array(composeImg)[x_ini:x_end, y_ini:y_end])
        patches.append(patchImg)

    return patches

#Example usage for patch generator
K = 5

rootDirEdges = "SK-LARGE/aug_data/ed_scale/1/o/0/f/0/"
rootDirSke = "SK-LARGE/aug_data/gt_scale/1/o/0/f/0/"

fileList = os.listdir(rootDirEdges)
fileNum = np.random.randint(len(fileList))
edgePath = rootDirEdges + fileList[fileNum]
skeletonPath = rootDirSke + fileList[fileNum]

patches = generatePatches(edgePath, skeletonPath, K)

edge = np.array(Image.open(edgePath))
ske = np.array(Image.open(skeletonPath))
composeImg = Image.fromarray(ske + edge)

fig = plt.figure(figsize=(15,5))
plt.clf()
plt.subplot(1,K + 1,1)
plt.imshow(composeImg)
for i in range(0,K):
    plt.subplot(1, K + 1, i + 2)
    plt.imshow(patches[i])
plt.show()