import torch
import torch.nn as nn

from models.dsconv import DSConv

activation_dict = {'sigmoid': nn.Sigmoid(), 'relu': nn.ReLU(True), 'leaky_relu': nn.LeakyReLU(inplace=True), 'relu6': nn.ReLU6(True)}


class YoloBackbone(nn.Module):
    def __init__(self):
        super(YoloBackbone, self).__init__()
        self.activation = activation_dict['leaky_relu']
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 7, 2, 3),
            nn.BatchNorm2d(32),
            self.activation,
        )
        self.conv2 = nn.Sequential(
            DSConv(32, 64, 3, 1, 1),
            DSConv(64, 48, 3, 1, 1),
            DSConv(48, 64, 3, 1, 1),
            self.maxpool
        )
        self.conv2_skip = nn.Sequential(
            nn.Conv2d(32, 64, 1, 1),
            nn.BatchNorm2d(64),
            self.activation,
            self.maxpool
        )
        self.conv3 = nn.Sequential(
            DSConv(64, 128, 3, 1, 1),
            DSConv(128, 96, 3, 1, 1),
            DSConv(96, 128, 3, 1, 1),
            self.maxpool
        )
        self.conv3_skip = nn.Sequential(
            nn.Conv2d(64, 128, 1, 1),
            nn.BatchNorm2d(128),
            self.activation,
            self.maxpool
        )
        self.conv4 = nn.Sequential(
            DSConv(128, 256, 3, 1, 1),
            DSConv(256, 192, 3, 1, 1),
            DSConv(192, 256, 3, 1, 1),
            DSConv(256, 192, 3, 1, 1),
            DSConv(192, 256, 3, 1, 1),
            self.maxpool
        )
        self.conv4_skip = nn.Sequential(
            nn.Conv2d(128, 256, 1, 1),
            nn.BatchNorm2d(256),
            self.activation,
            self.maxpool
        )
        self.conv5 = nn.Sequential(
            DSConv(256, 512, 3, 1, 1),
            *[nn.Sequential(DSConv(512, 384, 3, 1, 1), DSConv(384, 512, 3, 1, 1)) for _ in range(7)],
            self.maxpool
        )
        self.conv5_skip = nn.Sequential(
            nn.Conv2d(256, 512, 1, 1),
            nn.BatchNorm2d(512),
            self.activation,
            self.maxpool
        )
        self.conv6 = nn.Sequential(
            DSConv(512, 1024, 3, 1, 1),
            DSConv(1024, 768, 3, 1, 1),
            DSConv(768, 1024, 3, 1, 1),
            DSConv(1024, 768, 3, 1, 1),
            DSConv(768, 1024, 3, 1, 1)
        )
        self.conv6_skip = nn.Sequential(
            nn.Conv2d(512, 1024, 1, 1),
            nn.BatchNorm2d(1024),
            self.activation
        )

    def forward(self, x):
        x1 = self.conv1(x)

        x2 = self.conv2(x1)
        x2_skip = self.conv2_skip(x1)
        x2 = x2 + x2_skip

        x3 = self.conv3(x2)
        x3_skip = self.conv3_skip(x2)
        x3 = x3 + x3_skip

        x4 = self.conv4(x3)
        x4_skip = self.conv4_skip(x3)
        x4 = x4 + x4_skip

        x5 = self.conv5(x4)
        x5_skip = self.conv5_skip(x4)
        x5 = x5 + x5_skip

        x6 = self.conv6(x5)
        x6_skip = self.conv6_skip(x5)
        x6 = x6 + x6_skip

        return x6


if __name__ == '__main__':
    from torchsummary import summary
    model = YoloBackbone().cuda()
    summary(model, (3, 416, 416))
















