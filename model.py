import torch 
from torch.nn.functional import interpolate
from torch import sigmoid

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

def initialize_fsds(path):
    net = FSDS()
    return load_vgg16(net,path)

def initialize_lmsds(path):
    net = LMSDS()
    return load_vgg16(net,path)

class FSDS(torch.nn.Module):
    def __init__(self):
        super(FSDS, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.conv3 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.conv4 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.conv5 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
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
            kernel_size=1, stride=1, padding=0, bias=False)

        self.fuseScale1 = torch.nn.Conv2d(in_channels=4, out_channels=1,
            kernel_size=1, stride=1, padding=0, bias=False)

        self.fuseScale2 = torch.nn.Conv2d(in_channels=3, out_channels=1,
            kernel_size=1, stride=1, padding=0, bias=False)

        self.fuseScale3 = torch.nn.Conv2d(in_channels=2, out_channels=1,
            kernel_size=1, stride=1, padding=0, bias=False)
        
        self.softmax = torch.nn.Softmax(dim=1)
        
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

        height = image.size(2)
        width = image.size(3)
        
        sideOut2 = interpolate(self.sideOut2(conv2), size=(height,width), mode='bilinear', align_corners=False)
        sideOut3 = interpolate(self.sideOut3(conv3), size=(height,width), mode='bilinear', align_corners=False)
        sideOut4 = interpolate(self.sideOut4(conv4), size=(height,width), mode='bilinear', align_corners=False)
        sideOut5 = interpolate(self.sideOut5(conv5), size=(height,width), mode='bilinear', align_corners=False)

        softSideOut2 = self.softmax(sideOut2)
        softSideOut3 = self.softmax(sideOut3)
        softSideOut4 = self.softmax(sideOut4)
        softSideOut5 = self.softmax(sideOut5)

        fuse0 = torch.cat((softSideOut2[:,0:1,:,:], softSideOut3[:,0:1,:,:], softSideOut4[:,0:1,:,:], softSideOut5[:,0:1,:,:] ),1)
        fuse1 = torch.cat((softSideOut2[:,1:2,:,:], softSideOut3[:,1:2,:,:], softSideOut4[:,1:2,:,:], softSideOut5[:,1:2,:,:] ),1)
        fuse2 = torch.cat((softSideOut3[:,2:3,:,:], softSideOut4[:,2:3,:,:], softSideOut5[:,2:3,:,:] ),1)
        fuse3 = torch.cat((softSideOut4[:,3:4,:,:], softSideOut5[:,3:4,:,:] ),1)
        fuse4 = softSideOut5[:,4:5,:,:]
        
        fuse0 = self.fuseScale0(fuse0)
        fuse1 = self.fuseScale1(fuse1)
        fuse2 = self.fuseScale2(fuse2)
        fuse3 = self.fuseScale3(fuse3)
        
        fuse = torch.cat((fuse0,fuse1,fuse2,fuse3,fuse4),1)
        
        #we do not ouptut softmax funtions as they are calculated with the cross entropy loss

        return sideOut2, sideOut3, sideOut4, sideOut5, fuse


class LMSDS(torch.nn.Module):
    def __init__(self):
        super(LMSDS, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.conv3 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.conv4 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.conv5 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
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
            kernel_size=1, stride=1, padding=0, bias=False)

        self.fuseScale1 = torch.nn.Conv2d(in_channels=4, out_channels=1,
            kernel_size=1, stride=1, padding=0, bias=False)

        self.fuseScale2 = torch.nn.Conv2d(in_channels=3, out_channels=1,
            kernel_size=1, stride=1, padding=0, bias=False)

        self.fuseScale3 = torch.nn.Conv2d(in_channels=2, out_channels=1,
            kernel_size=1, stride=1, padding=0, bias=False)

        with torch.no_grad():

            self.fuseScale0.weight.div_(torch.norm(self.fuseScale0.weight, dim=1, keepdim=True))
            self.fuseScale1.weight.div_(torch.norm(self.fuseScale1.weight, dim=1, keepdim=True))
            self.fuseScale2.weight.div_(torch.norm(self.fuseScale2.weight, dim=1, keepdim=True))
            self.fuseScale3.weight.div_(torch.norm(self.fuseScale3.weight, dim=1, keepdim=True))
        
        self.softmax = torch.nn.Softmax(dim=1)
        
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

        height = image.size(2)
        width = image.size(3)
        
        sideOutLoc2 = interpolate(self.sideOutLoc2(conv2), size=(height,width), mode='bilinear', align_corners=False)
        sideOutLoc3 = interpolate(self.sideOutLoc3(conv3), size=(height,width), mode='bilinear', align_corners=False)
        sideOutLoc4 = interpolate(self.sideOutLoc4(conv4), size=(height,width), mode='bilinear', align_corners=False)
        sideOutLoc5 = interpolate(self.sideOutLoc5(conv5), size=(height,width), mode='bilinear', align_corners=False)

        softSideOut2 = self.softmax(sideOutLoc2)
        softSideOut3 = self.softmax(sideOutLoc3)
        softSideOut4 = self.softmax(sideOutLoc4)
        softSideOut5 = self.softmax(sideOutLoc5)

        fuse0 = torch.cat((softSideOut2[:,0:1,:,:], softSideOut3[:,0:1,:,:], softSideOut4[:,0:1,:,:], softSideOut5[:,0:1,:,:] ),1)
        fuse1 = torch.cat((softSideOut2[:,1:2,:,:], softSideOut3[:,1:2,:,:], softSideOut4[:,1:2,:,:], softSideOut5[:,1:2,:,:] ),1)
        fuse2 = torch.cat((softSideOut3[:,2:3,:,:], softSideOut4[:,2:3,:,:], softSideOut5[:,2:3,:,:] ),1)
        fuse3 = torch.cat((softSideOut4[:,3:4,:,:], softSideOut5[:,3:4,:,:] ),1)
        fuse4 = softSideOut5[:,4:5,:,:]
        
        fuse0 = self.fuseScale0(fuse0)
        fuse1 = self.fuseScale1(fuse1)
        fuse2 = self.fuseScale2(fuse2)
        fuse3 = self.fuseScale3(fuse3)
        
        fuse = torch.cat((fuse0,fuse1,fuse2,fuse3,fuse4),1)
        
        sideOutScale2 = interpolate(self.sideOutScale2(conv2), size=(height,width), mode='bilinear', align_corners=False)
        sideOutScale3 = interpolate(self.sideOutScale3(conv3), size=(height,width), mode='bilinear', align_corners=False)
        sideOutScale4 = interpolate(self.sideOutScale4(conv4), size=(height,width), mode='bilinear', align_corners=False)
        sideOutScale5 = interpolate(self.sideOutScale5(conv5), size=(height,width), mode='bilinear', align_corners=False)

        sideOutScale2 = sigmoid(sideOutScale2)
        sideOutScale3 = sigmoid(sideOutScale3)
        sideOutScale4 = sigmoid(sideOutScale4)
        sideOutScale5 = sigmoid(sideOutScale5)

        #we do not ouptut softmax funtions as they are calculated with the cross entropy loss

        return sideOutLoc2, sideOutLoc3, sideOutLoc4, sideOutLoc5, fuse, sideOutScale2, sideOutScale3, sideOutScale4, sideOutScale5