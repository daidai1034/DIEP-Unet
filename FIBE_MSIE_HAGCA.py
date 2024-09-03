from functools import partial
from models.module.HAGCA import *
from models.module.MSIE import *
from models.module.FIBE import *

nonlinearity = partial(F.relu, inplace=True)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class RB(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(RB, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)

    def forward(self, input):
        Y = self.conv(input)
        X = self.conv3(input)
        return Y+X


class RDB(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(RDB, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, dilation=1, padding=1),
            nn.BatchNorm2d(out_ch, ),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)

    def forward(self, input):
        Y = self.conv(input)
        X = self.conv3(input)
        return Y+X


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super(UNet, self).__init__()
        n = 16
        filters = [n, n * 2, n * 4, n * 8, n * 16]

        self.F = IBE()

        self.hgca_1 = HAGCA_1(filters[0], filters[0])
        self.hgca_2 = HAGCA_1(filters[1], filters[1])
        self.hgca_3 = HAGCA_1(filters[2], filters[2])
        self.hgca_4 = HAGCA_1(filters[3], filters[3])
        self.hgca_5 = HAGCA_2(filters[4], filters[4])

        self.conv1 = RDB(in_ch, filters[0])
        self.pool1 = nn.MaxPool2d(2)


        self.M1 = MSIE(in_ch, filters[0])
        self.conv2 = RDB(filters[0], filters[1])
        self.pool2 = nn.MaxPool2d(2)

        self.M2 = MSIE(in_ch, filters[1])
        self.conv3 = RDB(filters[1], filters[2])
        self.pool3 = nn.MaxPool2d(2)

        self.M3 = MSIE(in_ch, filters[2])
        self.conv4 = RDB(filters[2], filters[3])
        self.pool4 = nn.MaxPool2d(2)

        self.M4 = MSIE(in_ch, filters[3])
        self.conv5 = RDB(filters[3], filters[4])

        # center

        # decoder
        self.conv6 = RB(filters[4], filters[3])
        self.conv7 = RB(filters[3], filters[2])
        self.conv8 = RB(filters[2], filters[1])
        self.conv9 = RB(filters[1], filters[0])

        self.up6 = nn.ConvTranspose2d(filters[4], filters[3], 2, stride=2)
        self.up7 = nn.ConvTranspose2d(filters[3], filters[2], 2, stride=2)
        self.up8 = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)
        self.up9 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)

        self.out = conv1x1(filters[0], out_ch)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        h1 = self.hgca_1(c1)

        L1 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        f1 = self.F(L1)
        lf1 = f1 +L1
        MP1 = self.M1(lf1) + p1
        c2 = self.conv2(MP1)
        p2 = self.pool2(c2)
        h2 = self.hgca_2(c2)

        L2 = F.interpolate(L1, scale_factor=0.5, mode='bilinear')
        f2 = self.F(L2)
        lf2 = f2 + L2
        MP2 = self.M2(lf2) + p2
        c3 = self.conv3(MP2)
        p3 = self.pool3(c3)
        h3 = self.hgca_3(c3)

        L3 = F.interpolate(L2, scale_factor=0.5, mode='bilinear')
        f3 = self.F(L3)
        lf3 = f3 + L3
        MP3= self.M3(lf3) + p3
        c4 = self.conv4(MP3)
        p4 = self.pool4(c4)
        h4 = self.hgca_4(c4)

        L4 = F.interpolate(L3, scale_factor=0.5, mode='bilinear')
        f4 = self.F(L4)
        lf4 = f4 + L4
        MP4 = self.M4(lf4) + p4
        c5 = self.conv5(MP4)
        h5 = self.hgca_5(c5)   # center

        # decoding + concat path
        up_6 = self.up6(h5)
        merge6 = torch.cat([up_6, h4], dim=1)
        c6 = self.conv6(merge6)

        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, h3], dim=1)
        c7 = self.conv7(merge7)

        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, h2], dim=1)
        c8 = self.conv8(merge8)

        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, h1], dim=1)
        c9 = self.conv9(merge9)

        out = self.out(c9)
        return out












