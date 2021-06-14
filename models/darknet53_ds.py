import torch
import torch.nn as nn

from models.dsconv import DSConv
from torchvision.models import ResNet


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels):
        super(BottleneckBlock, self).__init__()
        mid_channels = int(in_channels / 2)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, 1, 0)
        self.conv2 = DSConv(mid_channels, in_channels, 3, 1, 1)
        self.conv_skip = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.lrelu = nn.LeakyReLU(.1, True)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        skip = x
        main = self.conv1(x)
        main = self.conv2(main)
        main = main.clone() + self.conv_skip(skip)
        # x = self.relu(main)

        return x


class Darknet53DS(nn.Module):
    def __init__(self):
        super(Darknet53DS, self).__init__()

        self.conv_1 = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU())
        self.conv_2 = nn.Sequential(DSConv(32, 64, 3, 2, 1), nn.ReLU())
        self.conv_4 = nn.Sequential(DSConv(64, 128, 3, 2, 1), nn.ReLU())
        self.conv_6 = nn.Sequential(DSConv(128, 256, 3, 2, 1), nn.ReLU())
        self.conv_8 = nn.Sequential(DSConv(256, 512, 3, 2, 1), nn.ReLU())
        self.conv_10 = nn.Sequential(DSConv(512, 1024, 3, 2, 1), nn.ReLU())
        self.conv_block_3 = self._make_layer(BottleneckBlock, 64, 1)
        self.conv_block_5 = self._make_layer(BottleneckBlock, 128, 2)
        self.conv_block_7 = self._make_layer(BottleneckBlock, 256, 8)
        self.conv_block_9 = self._make_layer(BottleneckBlock, 512, 8)
        self.conv_block_11 = self._make_layer(BottleneckBlock, 1024, 4)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_block_3(x)
        x = self.conv_4(x)
        x = self.conv_block_5(x)
        x = self.conv_6(x)
        x = self.conv_block_7(x)
        x = self.conv_8(x)
        x = self.conv_block_9(x)
        x = self.conv_10(x)
        x = self.conv_block_11(x)

        return x

    @staticmethod
    def _make_layer(block, in_channels, num_repeat):
        layers = []
        layers.append(block(in_channels))
        for i in range(1, num_repeat):
            layers.append(block(in_channels))

        return nn.Sequential(*layers)


if __name__ == '__main__':
    from torchsummary import summary
    model = Darknet53DS().cuda()
    summary(model, (3, 416, 416))

























