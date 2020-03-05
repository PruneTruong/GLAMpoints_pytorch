
import numpy as np
from .unet_parts import *
import torch.nn.functional as F
import torch


class UNet(nn.Module):
    def __init__(self, bilinear=False):
        super(UNet, self).__init__()
        n_channels = 1
        self.bilinear = bilinear

        # those names must correcpond to the names given to the different layers in the tensorflow model.
        self.conv_1 = double_conv(n_channels, 8)
        self.conv_2 = double_conv(8, 16)
        self.conv_3 = double_conv(16, 32)
        self.conv_4 = double_conv(32, 64)
        self.conv_5 = double_conv(64, 128) # conv_5 has 128 channels after 2 conv
        self.down = down()

        # up-scaling with deconv module
        self.decon_6 = up(128, 64, bilinear)
        self.decon_7 = up(64, 32, bilinear)
        self.decon_8 = up(32, 16, bilinear)
        self.decon_9 = up(16, 8, bilinear)

        # double conv
        self.conv_6 = double_conv(128, 64)
        self.conv_7 = double_conv(64, 32)
        self.conv_8 = double_conv(32, 16)
        self.conv_9 = double_conv(16, 8)

        self.final = nn.Conv2d(8, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data) # same initialisation than version using tensorflow
                if m.bias is not None:
                    m.bias.data.zero_()

    def concat(self, x1, x2):
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        upscale1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])

        x = torch.cat([upscale1, x2], dim=1)
        return x

    def forward(self, x):
        x1 = self.conv_1(x)
        x = self.down(x1)

        x2 = self.conv_2(x)
        x = self.down(x2)

        x3 = self.conv_3(x)
        x = self.down(x3)

        x4 = self.conv_4(x)
        x = self.down(x4)

        x5 = self.conv_5(x)

        x = self.decon_6(x5)
        x = self.concat(x, x4)
        x = self.conv_6(x)

        x = self.decon_7(x)
        x = self.concat(x, x3)
        x = self.conv_7(x)

        x = self.decon_8(x)
        x = self.concat(x, x2)
        x = self.conv_8(x)

        x = self.decon_9(x)
        x = self.concat(x, x1)
        x = self.conv_9(x)

        output = self.final(x)
        output_sigmoid = self.sigmoid(output)
        return output_sigmoid



class UNetWithClasses(nn.Module):
    def __init__(self, bilinear=False):
        super(UNetWithClasses, self).__init__()
        n_channels = 1
        self.bilinear = bilinear

        self.conv1 = Down(n_channels, 8)
        self.conv2 = Down(8, 16)
        self.conv3 = Down(16, 32)
        self.conv4 = Down(32, 64)
        self.conv5 = Down(64, 128) # conv_5 has 128 channels after 2 conv
        self.up1 = Up(128, 64, bilinear) # Up is upscaling (and reducing channel to 64) then concat and double conv
        self.up2 = Up(64, 32, bilinear)
        self.up3 = Up(32, 16, bilinear)
        self.up4 = Up(16, 8, bilinear)
        self.final = nn.Conv2d(8, 1, kernel_size=1, padding=0)
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
        output = self.final(x)
        output_sigmoid = self.sigmoid(output)
        return output_sigmoid