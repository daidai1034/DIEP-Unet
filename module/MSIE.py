from torch import nn
import torch.nn.functional as F

class MSIE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSIE, self).__init__()
        self.W_g_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
        )

        self.W_g_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.W_g_1(x)
        x2 = self.W_g_2(x)

        return F.relu(x1 + x2)