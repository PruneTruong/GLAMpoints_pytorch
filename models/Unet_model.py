
import numpy as np
from .unet_parts import *
from skimage.feature import peak_local_max


class UNet(nn.Module):
    def __init__(self, bilinear=False):
        super(UNet, self).__init__()
        n_channels = 1
        self.bilinear = bilinear

        self.conv1 = DoubleConv(n_channels, 8)
        self.conv2 = Down(8, 16)
        self.conv3 = Down(16, 32)
        self.conv4 = Down(32, 64)
        self.conv5 = Down(64, 128) # conv5 has 128 channels after 2 conv
        self.up1 = Up(128, 64, bilinear) # Up is upscaling (and reducing channel to 64) then concat and double conv
        self.up2 = Up(64, 32, bilinear)
        self.up3 = Up(32, 16, bilinear)
        self.up4 = Up(16, 8, bilinear)
        self.last_conv = nn.Conv2d(8, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data) # same initialisation than version using tensorflow
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        output = self.last_conv(x)
        output_sigmoid = self.sigmoid(output)
        return output_sigmoid
