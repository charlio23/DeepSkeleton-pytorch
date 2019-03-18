from dataset import SKLARGE
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def grayTrans(img):
    img = img.data.cpu().numpy()[0]*255.0
    img = (img).astype(np.uint8)
    img = Image.fromarray(img, 'L')
    return img

rootDir = "SK-LARGE/"
trainListPath = "SK-LARGE/aug_data/train_pair.lst"

trainDS = SKLARGE(rootDir, trainListPath)
train = DataLoader(trainDS, shuffle=True, batch_size=1, num_workers=4)

receptive_fields = np.array([14,40,92,196])
p = 1.2

def apply_quantization(scale):
    if scale < 0.001:
        return 0
    if p*scale > np.max(receptive_fields):
        return len(receptive_fields)
    return np.argmax(receptive_fields > p*scale) + 1

ite = iter(train)

while True:
    image, scale = ite.next()
    quantization = np.vectorize(lambda s: 0 if s < 0.001 else np.argmax(receptive_fields > p*s) + 1)
    quantise = torch.from_numpy(quantization(scale.numpy())).squeeze_(1)

    fig = plt.figure(figsize=(16,4))

    plt.subplot(1,6,1)
    plt.imshow(np.transpose(image[0].cpu().numpy(), (1, 2, 0)))

    selection = (quantise > 0.01).float()
    image = grayTrans(selection)
    plt.subplot(1, 6, 2)
    plt.imshow(image)


    for i in range(1,5):
        selection = (quantise == i).float()
        image = grayTrans(selection)
        plt.subplot(1, 6, i + 2)
        plt.imshow(image)

    plt.show()