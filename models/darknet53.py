import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.acti = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.acti(x)

        return x


class BottleneckResidualBlock(nn.Module):
    def __init__(self, channels, mid_channels):
        super().__init__()
        self.conv1 = Conv(channels, mid_channels, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(True)

    def forward(self, x):
        skip = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += skip
        x = self.relu(x)

        return x


class Darknet53(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            Conv(3, 32, 3, 1, 1),
            Conv(32, 64, 3, 2, 1)
        )
        self.layer2 = nn.Sequential(
            self._make_conv_blocks(64, 32, 1),
            Conv(64, 128, 3, 2, 1)
        )
        self.layer3 = nn.Sequential(
            self._make_conv_blocks(128, 64, 2),
            Conv(128, 256, 3, 2, 1)
        )
        self.layer4 = nn.Sequential(
            self._make_conv_blocks(256, 128, 8),
            Conv(256, 512, 3, 2, 1)
        )
        self.layer5 = nn.Sequential(
            self._make_conv_blocks(512, 256, 8),
            Conv(512, 1024, 3, 2, 1)
        )
        self.layer6 = self._make_conv_blocks(1024, 512, 4)

    def _make_conv_blocks(self, channels, mid_channels, num_repeat):
        return nn.Sequential(
            *[BottleneckResidualBlock(channels, mid_channels) for _ in range(num_repeat)]
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        return x












if __name__ == '__main__':
    from torchsummary import summary
    model = Darknet53().cuda()
    summary(model, (3, 416, 416))









































