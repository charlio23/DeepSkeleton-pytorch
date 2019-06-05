from PIL import Image, ImageDraw
import os
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import distance_transform_edt as bwdist

"""
outDir = "images-test/"
rootDirGt = "SK-LARGE/groundTruth/test/"
listData = sorted(os.listdir(rootDirGt))
os.makedirs(outDir,exist_ok=True)
for element in tqdm(listData):
    path = rootDirGt + element
    element = element.replace('.mat','.png')
    itemGround = loadmat(path)
    edge, skeleton = itemGround['edge'], itemGround['symmetry'].astype(float)
    skeleton[skeleton>0] = 255.0
    Image.fromarray(skeleton.astype(np.uint8),'L').save(outDir + element)
"""
outDir = "images-test-segmentation/"
rootDirGt = "SK-LARGE/groundTruth/test/"
listData = sorted(os.listdir(rootDirGt))
os.makedirs(outDir,exist_ok=True)
for element in tqdm(listData):
    if element != "b42b833d8f1fe2ed89a291bb481a0c88.mat":
        continue
    path = rootDirGt + element
    element = element.replace('.mat','.png')
    itemGround = loadmat(path)
    edge, skeleton = itemGround['edge'], itemGround['symmetry']
    dist = bwdist(1.0 - edge.astype(float))
    make_scale = np.vectorize(lambda x, y: 0 if y < 0.5 else x)
    scale = make_scale(dist,skeleton)
    indices = np.argwhere(skeleton)
    img = Image.new("L",(len(skeleton[0]),len(skeleton)))
    draw = ImageDraw.Draw(img)
    print(np.max(itemGround['edge']))
    plt.imshow(Image.fromarray((itemGround['edge']*255.0).astype(np.uint8),'L'))
    plt.show()
    plt.imshow(Image.fromarray((itemGround['symmetry']*255.0).astype(np.uint8),'L'))
    plt.show()
    for (i, j) in indices:
        r = scale[i,j]
        x = j
        y = i
        draw.ellipse((x-r, y-r, x+r, y+r), fill='white')
    plt.imshow(img)
    plt.show()
    img.save(outDir + element)

