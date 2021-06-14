import torch
import torch.nn as nn

from models.backbone import YoloBackbone
from models.darknet19_ds import Darknet19DS
from models.darknet53_ds import Darknet53DS
from models.darknet53 import *
from models.dsconv import DSConv
from models.darknet19 import *
from utils.rpn_util import generate_anchor_box


class YOLOV2(nn.Module):
    def __init__(self, in_size, num_classes):
        """
        :param in_size: tuple or list, (height of input, width of input)
        :param num_classes: int
        :param anchor_box_samples: tensor, [num anchor boxes, (height of anchor box, width of anchor box)]
        """
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.in_size = in_size
        self.num_classes = num_classes
        self.anchors = torch.Tensor([[1.73145, 1.3221],
                                       [4.00944, 3.19275],
                                       [8.09892, 5.05587],
                                       [4.84053, 9.47112],
                                       [10.0071, 11.2364]])
        self.anchor_boxes = self._generate_anchor_box(anchor_box_sizes=self.anchors,
                                                      out_size=(13, 13))

        self.feature_model = Darknet19()
        # self.reg_layer = DSConv(1024, len(anchor_box_samples) * (5 + num_classes), 1, 1, 0, 'relu')
        self.reg_layer = nn.Sequential(
            nn.Conv2d(1024, 5 * (5 + num_classes), 1, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _generate_anchor_box(self, anchor_box_sizes, out_size):
        """
        Make anchor box the same shape as output's.
        :param anchor_box_sizes: tensor, [# anchor box, (height, width)]
        :param out_size: tuple or list, (height, width)

        :return tensor, [height, width, (cy, cx, h, w) * num bounding box]
        """

        out = torch.zeros(out_size[0], out_size[1], 4 * len(anchor_box_sizes)).cuda()
        cy_ones = torch.ones(1, out_size[1])
        cx_ones = torch.ones(out_size[0], 1)
        cy_tensor = torch.zeros(1, out_size[1])
        cx_tensor = torch.zeros(out_size[0], 1)
        for i in range(1, out_size[0]):
            cy_tensor = torch.cat([cy_tensor, cy_ones * i], dim=0)
            cx_tensor = torch.cat([cx_tensor, cx_ones * i], dim=1)

        ctr_tensor = torch.cat([cy_tensor.unsqueeze(2), cx_tensor.unsqueeze(2)], dim=2)

        for i in range(len(anchor_box_sizes)):
            out[:, :, 4 * i:4 * i + 2] = ctr_tensor
            out[:, :, 4 * i + 2] = anchor_box_sizes[i, 0]
            out[:, :, 4 * i + 3] = anchor_box_sizes[i, 1]

        return out

    def _activate_output(self, deltas, anchor_boxes):
        """
        :param deltas: tensor, [batch, height, width, ((dy, dx, dh, dw, p) + class scores) * num anchor boxes]
        :param anchor_boxes: tensor, [height, width, (cy, cx, h, w) * num anchor boxes]

        :return: tensor, [height, width, ((cy, cx, h, w, p) + class scores) * num anchor boxes]
        """

        out = torch.zeros(deltas.shape).to(self.device)
        sigmoid = torch.nn.Sigmoid()
        softmax = torch.nn.Softmax(dim=2)

        coord_per_box = 4
        num_anchor_boxes = 5
        num_data_per_box = 5 + self.num_classes

        anchor_boxes = anchor_boxes.unsqueeze(0)

        for i in range(num_anchor_boxes):
            out[..., num_data_per_box * i: num_data_per_box * i + 2] = (sigmoid(
                deltas[..., num_data_per_box * i:num_data_per_box * i + 2]
            ) + anchor_boxes[..., coord_per_box * i:coord_per_box * i + 2])
            out[..., num_data_per_box * i + 2:num_data_per_box * i + 4] = (torch.exp(
                torch.tanh(deltas[..., num_data_per_box * i + 2:num_data_per_box * i + 4])
            ) * anchor_boxes[..., coord_per_box * i + 2:coord_per_box * i + 4])
            out[..., num_data_per_box * i + 4] = sigmoid(deltas[..., num_data_per_box * i + 4])
            out[..., num_data_per_box * i + 5:num_data_per_box * (i + 1)] = softmax(
                deltas[..., num_data_per_box * i + 5:num_data_per_box * (i + 1)]
            )

        return out

    def forward(self, x):
        x = self.feature_model(x)
        x = self.reg_layer(x)
        x = x.permute(0, 2, 3, 1)

        x = self._activate_output(x, self.anchor_boxes)

        return x


if __name__ == '__main__':
    from torchsummary import summary
    model = YOLOV2Mobile((416, 416), 92).cuda()
    summary(model, (3, 416, 416))
