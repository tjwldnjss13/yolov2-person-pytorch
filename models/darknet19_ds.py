import torch
import torch.nn as nn

from models.dsconv import DSConv


activation_dict = {'sigmoid': nn.Sigmoid(), 'relu': nn.ReLU(True), 'leaky_relu': nn.LeakyReLU(inplace=True), 'relu6': nn.ReLU6(True)}


class Darknet19DS(nn.Module):
    def __init__(self):
        super(Darknet19DS, self).__init__()
        self.activation_str = 'relu'
        self.activation = activation_dict[self.activation_str]
        self.layer_1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            DSConv(32, 64, 3, 1, 1, self.activation_str),
            nn.MaxPool2d(2, 2),

            DSConv(64, 128, 3, 1, 1, self.activation_str),
            # DSConv(128, 64, 1, 1, 0, self.activation_str),
            nn.Conv2d(128, 64, 1, 1, 0),
            nn.BatchNorm2d(64),
            self.activation,
            DSConv(64, 128, 3, 1, 1, self.activation_str),
            nn.MaxPool2d(2, 2),

            DSConv(128, 256, 3, 1, 1, self.activation_str),
            # DSConv(256, 128, 1, 1, 0, self.activation_str),
            nn.Conv2d(256, 128, 1, 1, 0),
            nn.BatchNorm2d(128),
            self.activation,
            DSConv(128, 256, 3, 1, 1, self.activation_str),
            nn.MaxPool2d(2, 2),

            DSConv(256, 512, 3, 1, 1, self.activation_str),
            # DSConv(512, 256, 1, 1, 0, self.activation_str),
            nn.Conv2d(512, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            self.activation,
            DSConv(256, 512, 3, 1, 1, self.activation_str),
            # DSConv(512, 256, 1, 1, 0, self.activation_str),
            nn.Conv2d(512, 256, 1, 1, 0),
            self.activation,
            DSConv(256, 512, 3, 1, 1, self.activation_str),
        )
        self.layer_2 = nn.Sequential(
            nn.MaxPool2d(2, 2),

            DSConv(512, 1024, 3, 1, 1, self.activation_str),
            # DSConv(1024, 512, 1, 1, 0, self.activation_str),
            nn.Conv2d(1024, 512, 1, 1, 0),
            self.activation,
            DSConv(512, 1024, 3, 1, 1, self.activation_str),
            # DSConv(1024, 512, 1, 1, 0, self.activation_str),
            nn.Conv2d(1024, 512, 1, 1, 0),
            nn.BatchNorm2d(512),
            self.activation,
            DSConv(512, 1024, 3, 1, 1, self.activation_str),
            DSConv(1024, 1024, 3, 1, 1, self.activation_str),
            DSConv(1024, 1024, 3, 1, 1, self.activation_str),
            DSConv(1024, 1024, 3, 1, 1, self.activation_str),
        )
        # self.layer_3 = DSConv(3072, 1024, 1, 1, 0, self.activation_str)
        self.layer_3 = nn.Sequential(
            nn.Conv2d(3072, 1024, 1, 1, 0),
            nn.BatchNorm2d(1024),
            self.activation
        )
        self.passthrough_layer = DSConv(512, 512, 3, 1, 1, self.activation_str)

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
        _pass = x = self.layer_1(x)
        x = self.layer_2(x)

        _pass = self.passthrough_layer(_pass)
        h, w = _pass.shape[2:]
        h_cut, w_cut = int(h / 2), int(w / 2)
        _pass = torch.cat([_pass[:, :, :h_cut, :w_cut],
                           _pass[:, :, :h_cut, w_cut:],
                           _pass[:, :, h_cut:, :w_cut],
                           _pass[:, :, h_cut:, w_cut:]], dim=1)

        x = torch.cat([x, _pass], dim=1)
        # x = self.activation(x)

        x = self.layer_3(x)

        return x


if __name__ == '__main__':
    from torchsummary import summary
    model = Darknet19DS().cuda()
    summary(model, (3, 416, 416))

















