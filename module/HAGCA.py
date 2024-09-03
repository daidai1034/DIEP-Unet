import torch
from torch import nn

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(cat)

        return x * self.sigmoid(out)

class ChannelAttention(nn.Module):
    def __init__(self, in_channel,out_channel):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channel,out_channel, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channel,out_channel, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = self.avg_pool(x)
        avg = self.fc1(avg)
        avg = self.relu1(avg)
        avg_out = self.fc2(avg)

        max = self.max_pool(x)
        max = self.fc1(max)
        max = self.relu1(max)
        max_out = self.fc2(max)

        out = avg_out + max_out
        return x * self.sigmoid(out)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class HAGCA_1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(HAGCA_1, self).__init__()
        self.relu = nn.ReLU(True)
        self.Spa = SpatialAttention()
        self.Cha = ChannelAttention(in_channel,out_channel)
        self.branch0 = BasicConv2d(in_channel, out_channel, 1)

        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        # self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x_spa = self.Spa(x)
        x_11 = self.branch0(x_spa)
        x_11_spa = x_11+x_spa
        x0 = self.Cha(x_11_spa)

        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + x0)
        return x

class HAGCA_2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(HAGCA_2, self).__init__()
        self.relu = nn.ReLU(True)
        self.Spa = SpatialAttention()
        self.Cha = ChannelAttention(in_channel,out_channel)
        self.branch0 = BasicConv2d(in_channel, out_channel, 1)

        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )

        self.conv_cat = BasicConv2d(3*out_channel, out_channel, 3, padding=1)
        # self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x_spa = self.Spa(x)
        x_11 = self.branch0(x_spa)
        x_11_spa = x_11 + x_spa
        x0 = self.Cha(x_11_spa)

        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2), 1))

        x = self.relu(x_cat + x0)
        return x