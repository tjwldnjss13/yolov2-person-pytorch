import torch
import torch.nn as nn

from models.conv import *


class Darknet19(nn.Module):
    def __init__(self):
        super(Darknet19, self).__init__()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.layer_1 = nn.Sequential(
            Conv(3, 32, 3, 1, 1),
        )
        self.layer_2 = nn.Sequential(
            self.maxpool,
            Conv(32, 64, 3, 1, 1)
        )
        self.layer_3 = nn.Sequential(
            self.maxpool,
            Conv(64, 128, 3, 1, 1),
            Conv(128, 64, 1),
            Conv(64, 128, 3, 1, 1)
        )
        self.layer_4 = nn.Sequential(
            self.maxpool,
            Conv(128, 256, 3, 1, 1),
            Conv(256, 128, 1),
            Conv(128, 256, 3, 1, 1)
        )
        self.layer_5 = nn.Sequential(
            self.maxpool,
            Conv(256, 512, 3, 1, 1),
            *[nn.Sequential(Conv(512, 256, 1), Conv(256, 512, 3, 1, 1)) for _ in range(2)],
        )
        self.layer_6 = nn.Sequential(
            self.maxpool,
            Conv(512, 1024, 3, 1, 1),
            *[nn.Sequential(Conv(1024, 512, 1), Conv(512, 1024, 3, 1, 1)) for _ in range(2)],
            *[Conv(1024, 1024, 3, 1, 1) for _ in range(2)]
        )
        self.layer_7 = Conv(3072, 1024, 3, 1, 1)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = skip = self.layer_5(x)
        x = self.layer_6(x)

        skip = skip.reshape(skip.shape[0], -1, x.shape[-2], x.shape[-1])
        x = torch.cat([x, skip], dim=1)

        x = self.layer_7(x)

        return x


if __name__ == '__main__':
    from torchsummary import summary
    model = Darknet19().cuda()
    summary(model, (3, 416, 416))







































