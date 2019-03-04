import torch 
from torch.nn.functional import interpolate
from torch import sigmoid

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        if m.in_channels == 5:
            torch.nn.init.constant_(m.weight.data,0.2)
        else:
            torch.nn.init.constant_(m.weight.data,0)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data,0)

def initialize_hed(path):
    net = HED()
    vgg16_items = list(torch.load(path).items())
    net.apply(weights_init)
    j = 0
    for k, v in net.state_dict().items():
        if k.find("conv") != -1:
            net.state_dict()[k].copy_(vgg16_items[j][1])
            j += 1
    return net

class HED(torch.nn.Module):
    def __init__(self):
        super(HED, self).__init__()

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

        self.sideOut1 = torch.nn.Conv2d(in_channels=64, out_channels=1,
            kernel_size=1, stride=1, padding=0)

        self.sideOut2 = torch.nn.Conv2d(in_channels=128, out_channels=1,
            kernel_size=1, stride=1, padding=0)

        self.sideOut3 = torch.nn.Conv2d(in_channels=256, out_channels=1,
            kernel_size=1, stride=1, padding=0)

        self.sideOut4 = torch.nn.Conv2d(in_channels=512, out_channels=1,
            kernel_size=1, stride=1, padding=0)

        self.sideOut5 = torch.nn.Conv2d(in_channels=512, out_channels=1,
            kernel_size=1, stride=1, padding=0)

        self.fuse = torch.nn.Conv2d(in_channels=5, out_channels=1,
            kernel_size=1, stride=1, padding=0)

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

        sideOut1 = interpolate(self.sideOut1(conv1), size=(height,width), mode='bilinear', align_corners=False)
        sideOut2 = interpolate(self.sideOut2(conv2), size=(height,width), mode='bilinear', align_corners=False)
        sideOut3 = interpolate(self.sideOut3(conv3), size=(height,width), mode='bilinear', align_corners=False)
        sideOut4 = interpolate(self.sideOut4(conv4), size=(height,width), mode='bilinear', align_corners=False)
        sideOut5 = interpolate(self.sideOut5(conv5), size=(height,width), mode='bilinear', align_corners=False)

        fuse = self.fuse(torch.cat((sideOut1, sideOut2, sideOut3, sideOut4, sideOut5), 1))

        sideOut1 = sigmoid(sideOut1)
        sideOut2 = sigmoid(sideOut2)
        sideOut3 = sigmoid(sideOut3)
        sideOut4 = sigmoid(sideOut4)
        sideOut5 = sigmoid(sideOut5)
        fuse = sigmoid(fuse)

        return sideOut1, sideOut2, sideOut3, sideOut4, sideOut5, fuse 