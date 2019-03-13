# DeepSkeleton Pytorch reimplementation

PyTorch reimplementation of [Deep Skeleton: Learning Multi-task Scale-associated Deep Side Outputs (LMSDS) for object skeleton extraction in natural images](https://arxiv.org/abs/1609.03659) using Python 3.6 and Pytorch 1.0.1

The main objective is to reproduce the algorithm as it is done in the official implementation using Pytorch.

### Contents

This repository contains two executable files:

- [main.py](https://github.com/charlio23/DeepSkeleton-pytorch/blob/master/main.py): Training scheme.
- [test_results.py](https://github.com/charlio23/DeepSkeleton-pytorch/blob/master/test_results.py): Code to generate test images.

### Data preprocessing and augmentation

Download [SK-LARGE](http://kaiz.xyz/sk-large) from the author's page and use their code to perform data augmentation.


### Training

Once you have your augmented data, you can try training the algorithm.

First you need to download the VGG16 pretrained model.

```
mkdir model
wget https://download.pytorch.org/models/vgg16-397923af.pth
mv vgg16-397923af.pth model/vgg16.pth
python main.py
```

### Test

Should you wish to test the algorithm:

```
python test_results.py
```

Use the [skeval](https://github.com/zeakey/skeval) code to preform test.

## References

- [DeepSkeleton: Learning Multi-task Scale-associated Deep Side Outputs for Object Skeleton Extraction in Natural Images](http://kaiz.xyz/deepsk) Shen, Wei and Zhao, Kai and Jiang, Yuan and Wang, Yan and Bai, Xiang and Yuille, Alan, IEEE Transactions on Image Processing, 2017, pp. 5298-5311.
