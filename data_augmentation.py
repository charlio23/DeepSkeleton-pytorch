from PIL import Image
import PIL
import os
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat

rootDirGt = "SK-LARGE/groundTruth/train/"
listData = sorted(os.listdir(rootDirGt))

output_dir = "SK-LARGE/aug_data/ed-scale/"
for scale in tqdm([0.8, 1.0, 1.2]):
    os.makedirs(output_dir + str(scale) + "/o/", exist_ok=True)
    for angle in tqdm([0, 90, 180, 270]):
        os.makedirs(output_dir + str(scale) + "/o/" + str(angle) + "/f/", exist_ok=True)
        for flip in tqdm([0, 1, 2]):
            os.makedirs(output_dir + str(scale) + "/o/" + str(angle) + "/f/" + str(flip), exist_ok=True)
            for targetName in tqdm(listData):
                path = output_dir + str(scale) + "/o/" + str(angle) + "/f/" + str(flip) + "/"
                name = targetName.replace(".mat", ".jpg")
                itemGround = loadmat(rootDirGt + targetName)
                edge = (itemGround['edge']).astype(float)*255.0
                img = (edge).astype(np.uint8)
                img = Image.fromarray(img, 'L')
                img = img.rotate(angle, PIL.Image.BICUBIC, True)
                new_size = (int(np.ceil(scale*img.size[0])), int(np.ceil(scale*img.size[1])))
                img = img.resize(new_size)
                if flip == 1:
                    img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                elif flip == 2:
                    img = img.transpose(PIL.Image.FLIP_TOP_BOTTOM)
                img.save(path + name)
