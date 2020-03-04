
import numpy as np
from .unet_parts import *
from skimage.feature import peak_local_max


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.conv1 = DoubleConv(n_channels, 8)
        self.conv2 = Down(8, 16)
        self.conv3 = Down(16, 32)
        self.conv4 = Down(32, 64)
        self.conv5 = Down(64, 128) # conv5 has 128 channels after 2 conv
        self.up1 = Up(128, 64, bilinear)
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


def non_max_suppression(image, size_filter, proba):
    non_max = peak_local_max(image, min_distance=size_filter, threshold_abs=proba, \
                      exclude_border=True, indices=False)
    kp = np.where(non_max>0)
    if len(kp[0]) != 0:
        for i in range(len(kp[0]) ):

            window=non_max[kp[0][i]-size_filter:kp[0][i]+(size_filter+1), \
                           kp[1][i]-size_filter:kp[1][i]+(size_filter+1)]
            if np.sum(window)>1:
                window[:,:]=0
    return non_max