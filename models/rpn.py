import torch
import numpy as np
import torch.nn as nn

from utils.util import calculate_ious
from utils.rpn_util import generate_anchor_box


class RPN(nn.Module):
    def __init__(self, in_dim, out_dim, in_size, n_anchor):
        super(RPN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_size = in_size
        self.conv = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.reg_layer = nn.Conv2d(out_dim, n_anchor * 4, 1, 1, 0)
        self.cls_layer = nn.Conv2d(out_dim, n_anchor, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()

        self._initialize_weights()

    def _initialize_weights(self):
        torch.nn.init.kaiming_uniform_(self.conv.weight)
        torch.nn.init.kaiming_uniform_(self.reg_layer.weight)
        torch.nn.init.kaiming_uniform_(self.cls_layer.weight)

    def forward(self, x):
        x = self.conv(x)

        reg = self.reg_layer(x)
        cls = self.cls_layer(x)

        # reg = reg.permute(0, 2, 3, 1).contiguous().view(reg.size(0), -1, 4)
        # cls = cls.permute(0, 2, 3, 1).contiguous().view(cls.size(0), -1, 2)

        cls = self.sigmoid(cls)

        return reg, cls


# if __name__ == '__main__':
#     import cv2 as cv
#     import matplotlib.pyplot as plt
#     import copy
#
#     rpn = RPN(512, 512, (14, 14), 9).cuda()
#     from torchsummary import summary
#     summary(rpn, (512, 14, 14))
#
#     ratios = [.5, 1, 2]
#     scales = [128, 256, 512]
#     in_size = (600, 1000)
#     anchor_boxes, valid_mask = generate_anchor_box(ratios, scales, in_size, 16)
#     valid_anchor_boxes = anchor_boxes[torch.nonzero(valid_mask, as_tuple=False)].squeeze(1)
#
#     img_pth = '../sample/dogs.jpg'
#     img = cv.imread(img_pth)
#     img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#     img_h_og, img_w_og, _ = img.shape
#     img = cv.resize(img, (in_size[1], in_size[0]))
#
#     bbox = np.array([[120, 70, 570, 280], [220, 270, 580, 450], [30, 440, 570, 700]])
#     bbox[:, 0] = bbox[:, 0] * (in_size[0] / img_h_og)
#     bbox[:, 1] = bbox[:, 1] * (in_size[1] / img_w_og)
#     bbox[:, 2] = bbox[:, 2] * (in_size[0] / img_h_og)
#     bbox[:, 3] = bbox[:, 3] * (in_size[1] / img_w_og)
#
#     img_copy = copy.deepcopy(img)
#
#     for i, box in enumerate(valid_anchor_boxes):
#         if i < 10:
#             print(box)
#         y1, x1, y2, x2 = box
#         y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)
#         cv.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 1)
#
#     for i, gt in enumerate(bbox):
#         y1, x1, y2, x2 = int(gt[0]), int(gt[1]), int(gt[2]), int(gt[3])
#         cv.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
#     plt.figure(figsize=(10, 6))
#     plt.imshow(img_copy)
#     plt.show()