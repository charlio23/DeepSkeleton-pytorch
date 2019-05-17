import torch 
from torch.nn.functional import interpolate
from torch import sigmoid
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        if m.in_channels < 5:
            n = float(m.in_channels)
            torch.nn.init.constant_(m.weight.data,1/n)
        else:
            torch.nn.init.constant_(m.weight.data,0)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data,0)

def load_vgg16(net, path):
    vgg16_items = list(torch.load(path).items())
    net.apply(weights_init)
    j = 0
    for k, v in net.state_dict().items():
        if k.find("conv") != -1:
            net.state_dict()[k].copy_(vgg16_items[j][1])
            j += 1
    return net

def load_checkpoint(net, path):
    dic = torch.load(path)
    dicli = list(dic.keys())
    new = {}
    j = 0
    for k in net.state_dict():
        if "Scale" in k:
            continue
        new[k] = dic[dicli[j]]
        j += 1
    net.load_state_dict(new)
    return net

def initialize_fsds(path,continue_train, path_HED=None):
    net = FSDS()
    if continue_train:
        return load_checkpoint(net,path_HED)
    return load_vgg16(net,path)

def initialize_lmsds(path,continue_train, path_HED=None):
    net = LMSDS()
    if continue_train:
        return load_checkpoint(net,path_HED)
    return load_vgg16(net,path)

class FSDS(torch.nn.Module):
    def __init__(self):
        super(FSDS, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,
                stride=1, padding=35),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU()
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU()
        )

        self.conv3 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU()
        )

        self.conv4 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU()
        )

        self.conv5 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU()
        )

        self.sideOut2 = torch.nn.Conv2d(in_channels=128, out_channels=2,
            kernel_size=1, stride=1, padding=0)

        self.sideOut3 = torch.nn.Conv2d(in_channels=256, out_channels=3,
            kernel_size=1, stride=1, padding=0)

        self.sideOut4 = torch.nn.Conv2d(in_channels=512, out_channels=4,
            kernel_size=1, stride=1, padding=0)

        self.sideOut5 = torch.nn.Conv2d(in_channels=512, out_channels=5,
            kernel_size=1, stride=1, padding=0)

        self.fuseScale0 = torch.nn.Conv2d(in_channels=4, out_channels=1,
            kernel_size=1, stride=1, padding=0)

        self.fuseScale1 = torch.nn.Conv2d(in_channels=4, out_channels=1,
            kernel_size=1, stride=1, padding=0)

        self.fuseScale2 = torch.nn.Conv2d(in_channels=3, out_channels=1,
            kernel_size=1, stride=1, padding=0)

        self.fuseScale3 = torch.nn.Conv2d(in_channels=2, out_channels=1,
            kernel_size=1, stride=1, padding=0)

        self.fuseScale4 = torch.nn.Conv2d(in_channels=1, out_channels=1,
            kernel_size=1, stride=1, padding=0)

        # Fixed bilinear weights.
        self.weight_deconv2 = make_bilinear_weights(4, 2)
        self.weight_deconv3 = make_bilinear_weights(8, 3)
        self.weight_deconv4 = make_bilinear_weights(16, 4)
        self.weight_deconv5 = make_bilinear_weights(32, 5)

        # Prepare for aligned crop.
        self.crop1_margin, self.crop2_margin, self.crop3_margin, self.crop4_margin, self.crop5_margin = \
            self.prepare_aligned_crop()

    # noinspection PyMethodMayBeStatic
    def prepare_aligned_crop(self):
        """ Prepare for aligned crop. """
        # Re-implement the logic in deploy.prototxt and
        #   /hed/src/caffe/layers/crop_layer.cpp of official repo.
        # Other reference materials:
        #   hed/include/caffe/layer.hpp
        #   hed/include/caffe/vision_layers.hpp
        #   hed/include/caffe/util/coords.hpp
        #   https://groups.google.com/forum/#!topic/caffe-users/YSRYy7Nd9J8

        def map_inv(m):
            """ Mapping inverse. """
            a, b = m
            return 1 / a, -b / a

        def map_compose(m1, m2):
            """ Mapping compose. """
            a1, b1 = m1
            a2, b2 = m2
            return a1 * a2, a1 * b2 + b1

        def deconv_map(kernel_h, stride_h, pad_h):
            """ Deconvolution coordinates mapping. """
            return stride_h, (kernel_h - 1) / 2 - pad_h

        def conv_map(kernel_h, stride_h, pad_h):
            """ Convolution coordinates mapping. """
            return map_inv(deconv_map(kernel_h, stride_h, pad_h))

        def pool_map(kernel_h, stride_h, pad_h):
            """ Pooling coordinates mapping. """
            return conv_map(kernel_h, stride_h, pad_h)

        x_map = (1, 0)
        conv1_1_map = map_compose(conv_map(3, 1, 35), x_map)
        conv1_2_map = map_compose(conv_map(3, 1, 1), conv1_1_map)
        pool1_map = map_compose(pool_map(2, 2, 0), conv1_2_map)

        conv2_1_map = map_compose(conv_map(3, 1, 1), pool1_map)
        conv2_2_map = map_compose(conv_map(3, 1, 1), conv2_1_map)
        pool2_map = map_compose(pool_map(2, 2, 0), conv2_2_map)

        conv3_1_map = map_compose(conv_map(3, 1, 1), pool2_map)
        conv3_2_map = map_compose(conv_map(3, 1, 1), conv3_1_map)
        conv3_3_map = map_compose(conv_map(3, 1, 1), conv3_2_map)
        pool3_map = map_compose(pool_map(2, 2, 0), conv3_3_map)

        conv4_1_map = map_compose(conv_map(3, 1, 1), pool3_map)
        conv4_2_map = map_compose(conv_map(3, 1, 1), conv4_1_map)
        conv4_3_map = map_compose(conv_map(3, 1, 1), conv4_2_map)
        pool4_map = map_compose(pool_map(2, 2, 0), conv4_3_map)

        conv5_1_map = map_compose(conv_map(3, 1, 1), pool4_map)
        conv5_2_map = map_compose(conv_map(3, 1, 1), conv5_1_map)
        conv5_3_map = map_compose(conv_map(3, 1, 1), conv5_2_map)

        score_dsn1_map = conv1_2_map
        score_dsn2_map = conv2_2_map
        score_dsn3_map = conv3_3_map
        score_dsn4_map = conv4_3_map
        score_dsn5_map = conv5_3_map

        upsample2_map = map_compose(deconv_map(4, 2, 0), score_dsn2_map)
        upsample3_map = map_compose(deconv_map(8, 4, 0), score_dsn3_map)
        upsample4_map = map_compose(deconv_map(16, 8, 0), score_dsn4_map)
        upsample5_map = map_compose(deconv_map(32, 16, 0), score_dsn5_map)

        crop1_margin = int(score_dsn1_map[1])
        crop2_margin = int(upsample2_map[1])
        crop3_margin = int(upsample3_map[1])
        crop4_margin = int(upsample4_map[1])
        crop5_margin = int(upsample5_map[1])

        return crop1_margin, crop2_margin, crop3_margin, crop4_margin, crop5_margin
        
    def forward(self, image):

        tensorBlue = (image[:, 0:1, :, :] * 255.0) - 104.00698793
        tensorGreen = (image[:, 1:2, :, :] * 255.0) - 116.66876762
        tensorRed = (image[:, 2:3, :, :] * 255.0) - 122.67891434

        image = torch.cat([ tensorBlue, tensorGreen, tensorRed ], 1)

        conv1 = self.conv1(image)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        image_h = image.size(2)
        image_w = image.size(3)

        score_dsn2 = self.sideOut2(conv2)
        score_dsn3 = self.sideOut3(conv3)
        score_dsn4 = self.sideOut4(conv4)
        score_dsn5 = self.sideOut5(conv5)

        upsample2 = torch.nn.functional.conv_transpose2d(score_dsn2, self.weight_deconv2, stride=2)
        upsample3 = torch.nn.functional.conv_transpose2d(score_dsn3, self.weight_deconv3, stride=4)
        upsample4 = torch.nn.functional.conv_transpose2d(score_dsn4, self.weight_deconv4, stride=8)
        upsample5 = torch.nn.functional.conv_transpose2d(score_dsn5, self.weight_deconv5, stride=16)

        # Aligned cropping.
        sideOut2 = upsample2[:, :, self.crop2_margin:self.crop2_margin+image_h,
                                self.crop2_margin:self.crop2_margin+image_w]
        sideOut3 = upsample3[:, :, self.crop3_margin:self.crop3_margin+image_h,
                                self.crop3_margin:self.crop3_margin+image_w]
        sideOut4 = upsample4[:, :, self.crop4_margin:self.crop4_margin+image_h,
                                self.crop4_margin:self.crop4_margin+image_w]
        sideOut5 = upsample5[:, :, self.crop5_margin:self.crop5_margin+image_h,
                                self.crop5_margin:self.crop5_margin+image_w]

        
        #sideOut2 = interpolate(self.sideOut2(conv2), size=(height,width), mode='bilinear', align_corners=False)
        #sideOut3 = interpolate(self.sideOut3(conv3), size=(height,width), mode='bilinear', align_corners=False)
        #sideOut4 = interpolate(self.sideOut4(conv4), size=(height,width), mode='bilinear', align_corners=False)
        #sideOut5 = interpolate(self.sideOut5(conv5), size=(height,width), mode='bilinear', align_corners=False)

        #softSideOut2 = self.softmax(sideOut2)
        #softSideOut3 = self.softmax(sideOut3)
        #softSideOut4 = self.softmax(sideOut4)
        #softSideOut5 = self.softmax(sideOut5)

        fuse0 = torch.cat((sideOut2[:,0:1,:,:], sideOut3[:,0:1,:,:], sideOut4[:,0:1,:,:], sideOut5[:,0:1,:,:] ),1)
        fuse1 = torch.cat((sideOut2[:,1:2,:,:], sideOut3[:,1:2,:,:], sideOut4[:,1:2,:,:], sideOut5[:,1:2,:,:] ),1)
        fuse2 = torch.cat((sideOut3[:,2:3,:,:], sideOut4[:,2:3,:,:], sideOut5[:,2:3,:,:] ),1)
        fuse3 = torch.cat((sideOut4[:,3:4,:,:], sideOut5[:,3:4,:,:] ),1)
        fuse4 = sideOut5[:,4:5,:,:]
        
        fuse0 = self.fuseScale0(fuse0)
        fuse1 = self.fuseScale1(fuse1)
        fuse2 = self.fuseScale2(fuse2)
        fuse3 = self.fuseScale3(fuse3)
        fuse4 = self.fuseScale4(fuse4)

        fuse = torch.cat((fuse0,fuse1,fuse2,fuse3,fuse4),1)
        
        #we do not ouptut softmax funtions as they are calculated with the cross entropy loss

        return sideOut2, sideOut3, sideOut4, sideOut5, fuse


class LMSDS(torch.nn.Module):
    def __init__(self):
        super(LMSDS, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,
                stride=1, padding=35),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU()
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU()
        )

        self.conv3 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU()
        )

        self.conv4 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU()
        )

        self.conv5 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU()
        )

        self.sideOutLoc2 = torch.nn.Conv2d(in_channels=128, out_channels=2,
            kernel_size=1, stride=1, padding=0)

        self.sideOutLoc3 = torch.nn.Conv2d(in_channels=256, out_channels=3,
            kernel_size=1, stride=1, padding=0)

        self.sideOutLoc4 = torch.nn.Conv2d(in_channels=512, out_channels=4,
            kernel_size=1, stride=1, padding=0)

        self.sideOutLoc5 = torch.nn.Conv2d(in_channels=512, out_channels=5,
            kernel_size=1, stride=1, padding=0)

        self.sideOutScale2 = torch.nn.Conv2d(in_channels=128, out_channels=1,
            kernel_size=1, stride=1, padding=0)

        self.sideOutScale3 = torch.nn.Conv2d(in_channels=256, out_channels=1,
            kernel_size=1, stride=1, padding=0)

        self.sideOutScale4 = torch.nn.Conv2d(in_channels=512, out_channels=1,
            kernel_size=1, stride=1, padding=0)

        self.sideOutScale5 = torch.nn.Conv2d(in_channels=512, out_channels=1,
            kernel_size=1, stride=1, padding=0)

        self.fuseScale0 = torch.nn.Conv2d(in_channels=4, out_channels=1,
            kernel_size=1, stride=1, padding=0)

        self.fuseScale1 = torch.nn.Conv2d(in_channels=4, out_channels=1,
            kernel_size=1, stride=1, padding=0)

        self.fuseScale2 = torch.nn.Conv2d(in_channels=3, out_channels=1,
            kernel_size=1, stride=1, padding=0)

        self.fuseScale3 = torch.nn.Conv2d(in_channels=2, out_channels=1,
            kernel_size=1, stride=1, padding=0)

        self.fuseScale4 = torch.nn.Conv2d(in_channels=1, out_channels=1,
            kernel_size=1, stride=1, padding=0)

        # Fixed bilinear weights.
        self.weight_deconv2 = make_bilinear_weights(4, 2)
        self.weight_deconv3 = make_bilinear_weights(8, 3)
        self.weight_deconv4 = make_bilinear_weights(16, 4)
        self.weight_deconv5 = make_bilinear_weights(32, 5)

        self.weight_deconvScale2 = make_bilinear_weights(4, 1)
        self.weight_deconvScale3 = make_bilinear_weights(8, 1)
        self.weight_deconvScale4 = make_bilinear_weights(16, 1)
        self.weight_deconvScale5 = make_bilinear_weights(32, 1)

        # Prepare for aligned crop.
        self.crop1_margin, self.crop2_margin, self.crop3_margin, self.crop4_margin, self.crop5_margin = \
            self.prepare_aligned_crop()

    # noinspection PyMethodMayBeStatic
    def prepare_aligned_crop(self):
        """ Prepare for aligned crop. """
        # Re-implement the logic in deploy.prototxt and
        #   /hed/src/caffe/layers/crop_layer.cpp of official repo.
        # Other reference materials:
        #   hed/include/caffe/layer.hpp
        #   hed/include/caffe/vision_layers.hpp
        #   hed/include/caffe/util/coords.hpp
        #   https://groups.google.com/forum/#!topic/caffe-users/YSRYy7Nd9J8

        def map_inv(m):
            """ Mapping inverse. """
            a, b = m
            return 1 / a, -b / a

        def map_compose(m1, m2):
            """ Mapping compose. """
            a1, b1 = m1
            a2, b2 = m2
            return a1 * a2, a1 * b2 + b1

        def deconv_map(kernel_h, stride_h, pad_h):
            """ Deconvolution coordinates mapping. """
            return stride_h, (kernel_h - 1) / 2 - pad_h

        def conv_map(kernel_h, stride_h, pad_h):
            """ Convolution coordinates mapping. """
            return map_inv(deconv_map(kernel_h, stride_h, pad_h))

        def pool_map(kernel_h, stride_h, pad_h):
            """ Pooling coordinates mapping. """
            return conv_map(kernel_h, stride_h, pad_h)

        x_map = (1, 0)
        conv1_1_map = map_compose(conv_map(3, 1, 35), x_map)
        conv1_2_map = map_compose(conv_map(3, 1, 1), conv1_1_map)
        pool1_map = map_compose(pool_map(2, 2, 0), conv1_2_map)

        conv2_1_map = map_compose(conv_map(3, 1, 1), pool1_map)
        conv2_2_map = map_compose(conv_map(3, 1, 1), conv2_1_map)
        pool2_map = map_compose(pool_map(2, 2, 0), conv2_2_map)

        conv3_1_map = map_compose(conv_map(3, 1, 1), pool2_map)
        conv3_2_map = map_compose(conv_map(3, 1, 1), conv3_1_map)
        conv3_3_map = map_compose(conv_map(3, 1, 1), conv3_2_map)
        pool3_map = map_compose(pool_map(2, 2, 0), conv3_3_map)

        conv4_1_map = map_compose(conv_map(3, 1, 1), pool3_map)
        conv4_2_map = map_compose(conv_map(3, 1, 1), conv4_1_map)
        conv4_3_map = map_compose(conv_map(3, 1, 1), conv4_2_map)
        pool4_map = map_compose(pool_map(2, 2, 0), conv4_3_map)

        conv5_1_map = map_compose(conv_map(3, 1, 1), pool4_map)
        conv5_2_map = map_compose(conv_map(3, 1, 1), conv5_1_map)
        conv5_3_map = map_compose(conv_map(3, 1, 1), conv5_2_map)

        score_dsn1_map = conv1_2_map
        score_dsn2_map = conv2_2_map
        score_dsn3_map = conv3_3_map
        score_dsn4_map = conv4_3_map
        score_dsn5_map = conv5_3_map

        upsample2_map = map_compose(deconv_map(4, 2, 0), score_dsn2_map)
        upsample3_map = map_compose(deconv_map(8, 4, 0), score_dsn3_map)
        upsample4_map = map_compose(deconv_map(16, 8, 0), score_dsn4_map)
        upsample5_map = map_compose(deconv_map(32, 16, 0), score_dsn5_map)

        crop1_margin = int(score_dsn1_map[1])
        crop2_margin = int(upsample2_map[1])
        crop3_margin = int(upsample3_map[1])
        crop4_margin = int(upsample4_map[1])
        crop5_margin = int(upsample5_map[1])

        return crop1_margin, crop2_margin, crop3_margin, crop4_margin, crop5_margin

    def forward(self, image):

        tensorBlue = (image[:, 0:1, :, :] * 255.0) - 104.00698793
        tensorGreen = (image[:, 1:2, :, :] * 255.0) - 116.66876762
        tensorRed = (image[:, 2:3, :, :] * 255.0) - 122.67891434

        image = torch.cat([ tensorBlue, tensorGreen, tensorRed ], 1)

        conv1 = self.conv1(image)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        image_h = image.size(2)
        image_w = image.size(3)

        score_dsn2 = self.sideOutLoc2(conv2)
        score_dsn3 = self.sideOutLoc3(conv3)
        score_dsn4 = self.sideOutLoc4(conv4)
        score_dsn5 = self.sideOutLoc5(conv5)

        upsample2 = torch.nn.functional.conv_transpose2d(score_dsn2, self.weight_deconv2, stride=2)
        upsample3 = torch.nn.functional.conv_transpose2d(score_dsn3, self.weight_deconv3, stride=4)
        upsample4 = torch.nn.functional.conv_transpose2d(score_dsn4, self.weight_deconv4, stride=8)
        upsample5 = torch.nn.functional.conv_transpose2d(score_dsn5, self.weight_deconv5, stride=16)

        # Aligned cropping.
        sideOut2 = upsample2[:, :, self.crop2_margin:self.crop2_margin+image_h,
                                self.crop2_margin:self.crop2_margin+image_w]
        sideOut3 = upsample3[:, :, self.crop3_margin:self.crop3_margin+image_h,
                                self.crop3_margin:self.crop3_margin+image_w]
        sideOut4 = upsample4[:, :, self.crop4_margin:self.crop4_margin+image_h,
                                self.crop4_margin:self.crop4_margin+image_w]
        sideOut5 = upsample5[:, :, self.crop5_margin:self.crop5_margin+image_h,
                                self.crop5_margin:self.crop5_margin+image_w]


        fuse0 = torch.cat((sideOut2[:,0:1,:,:], sideOut3[:,0:1,:,:], sideOut4[:,0:1,:,:], sideOut5[:,0:1,:,:] ),1)
        fuse1 = torch.cat((sideOut2[:,1:2,:,:], sideOut3[:,1:2,:,:], sideOut4[:,1:2,:,:], sideOut5[:,1:2,:,:] ),1)
        fuse2 = torch.cat((sideOut3[:,2:3,:,:], sideOut4[:,2:3,:,:], sideOut5[:,2:3,:,:] ),1)
        fuse3 = torch.cat((sideOut4[:,3:4,:,:], sideOut5[:,3:4,:,:] ),1)
        fuse4 = sideOut5[:,4:5,:,:]
        
        fuse0 = self.fuseScale0(fuse0)
        fuse1 = self.fuseScale1(fuse1)
        fuse2 = self.fuseScale2(fuse2)
        fuse3 = self.fuseScale3(fuse3)
        
        fuse = torch.cat((fuse0,fuse1,fuse2,fuse3,fuse4),1)
        
        score_dsnScale2 = self.sideOutScale2(conv2)
        score_dsnScale3 = self.sideOutScale3(conv3)
        score_dsnScale4 = self.sideOutScale4(conv4)
        score_dsnScale5 = self.sideOutScale5(conv5)

        upsampleScale2 = torch.nn.functional.conv_transpose2d(score_dsnScale2, self.weight_deconvScale2, stride=2)
        upsampleScale3 = torch.nn.functional.conv_transpose2d(score_dsnScale3, self.weight_deconvScale3, stride=4)
        upsampleScale4 = torch.nn.functional.conv_transpose2d(score_dsnScale4, self.weight_deconvScale4, stride=8)
        upsampleScale5 = torch.nn.functional.conv_transpose2d(score_dsnScale5, self.weight_deconvScale5, stride=16)

        # Aligned cropping.
        sideOutScale2 = upsample2[:, :, self.crop2_margin:self.crop2_margin+image_h,
                                self.crop2_margin:self.crop2_margin+image_w]
        sideOutScale3 = upsample3[:, :, self.crop3_margin:self.crop3_margin+image_h,
                                self.crop3_margin:self.crop3_margin+image_w]
        sideOutScale4 = upsample4[:, :, self.crop4_margin:self.crop4_margin+image_h,
                                self.crop4_margin:self.crop4_margin+image_w]
        sideOutScale5 = upsample5[:, :, self.crop5_margin:self.crop5_margin+image_h,
                                self.crop5_margin:self.crop5_margin+image_w]

        #we do not ouptut softmax funtions as they are calculated with the cross entropy loss

        return sideOut2, sideOut3, sideOut4, sideOut5, fuse, sideOutScale2, sideOutScale3, sideOutScale4, sideOutScale5

def make_bilinear_weights(size, num_channels):
    """ Generate bi-linear interpolation weights as up-sampling filters (following FCN paper). """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    filt = torch.from_numpy(filt)
    w = torch.zeros(num_channels, num_channels, size, size)
    w.requires_grad = False  # Set not trainable.
    for i in range(num_channels):
        for j in range(num_channels):
            if i == j:
                w[i, j] = filt
    return w.cuda()