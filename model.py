import torch 
from torch.nn.functional import interpolate

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
        
def initialize_fsds(path):
    net = FSDS()
    vgg16_items = list(torch.load(path).items())
    net.apply(weights_init)
    j = 0
    for k, v in net.state_dict().items():
        print(k)
        if k.find("conv") != -1:
            net.state_dict()[k].copy_(vgg16_items[j][1])
            j += 1
    return net

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
            kernel_size=1, stride=1, padding=0)

        self.fuseScale1 = torch.nn.Conv2d(in_channels=4, out_channels=1,
            kernel_size=1, stride=1, padding=0)

        self.fuseScale2 = torch.nn.Conv2d(in_channels=3, out_channels=1,
            kernel_size=1, stride=1, padding=0)

        self.fuseScale3 = torch.nn.Conv2d(in_channels=2, out_channels=1,
            kernel_size=1, stride=1, padding=0)
        
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