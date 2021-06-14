import torch
import torch.nn as nn

from utils.pytorch_util import calculate_iou
from utils.pytorch_util import convert_box_from_hw_to_yx

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# class YoloLoss(nn.modules.loss._Loss):
#     # The loss I borrow from LightNet repo.
#     def __init__(self, num_classes, anchors, reduction=32, coord_scale=1.0, noobject_scale=1.0,
#                  object_scale=5.0, class_scale=1.0, thresh=0.6):
#         super(YoloLoss, self).__init__()
#         self.num_classes = num_classes
#         self.num_anchors = len(anchors)
#         self.anchor_step = len(anchors[0])
#         self.anchors = torch.Tensor(anchors)
#         self.reduction = reduction
#
#         self.coord_scale = coord_scale
#         self.noobject_scale = noobject_scale
#         self.object_scale = object_scale
#         self.class_scale = class_scale
#         self.thresh = thresh
#
#     def forward(self, output, target):
#
#         batch_size = output.data.size(0)
#         height = output.data.size(2)
#         width = output.data.size(3)
#
#         # Get x,y,w,h,conf,cls
#         output = output.view(batch_size, self.num_anchors, -1, height * width)
#         coord = torch.zeros_like(output[:, :, :4, :])
#         coord[:, :, :2, :] = output[:, :, :2, :].sigmoid()
#         coord[:, :, 2:4, :] = output[:, :, 2:4, :]
#         conf = output[:, :, 4, :].sigmoid()
#         cls = output[:, :, 5:, :].contiguous().view(batch_size * self.num_anchors, self.num_classes,
#                                                     height * width).transpose(1, 2).contiguous().view(-1,
#                                                                                                       self.num_classes)
#
#         # Create prediction boxes
#         pred_boxes = torch.FloatTensor(batch_size * self.num_anchors * height * width, 4)
#         lin_x = torch.range(0, width - 1).repeat(height, 1).view(height * width)
#         lin_y = torch.range(0, height - 1).repeat(width, 1).t().contiguous().view(height * width)
#         anchor_w = self.anchors[:, 0].contiguous().view(self.num_anchors, 1)
#         anchor_h = self.anchors[:, 1].contiguous().view(self.num_anchors, 1)
#
#         if torch.cuda.is_available():
#             pred_boxes = pred_boxes.cuda()
#             lin_x = lin_x.cuda()
#             lin_y = lin_y.cuda()
#             anchor_w = anchor_w.cuda()
#             anchor_h = anchor_h.cuda()
#
#         pred_boxes[:, 0] = (coord[:, :, 0].detach() + lin_x).view(-1)
#         pred_boxes[:, 1] = (coord[:, :, 1].detach() + lin_y).view(-1)
#         pred_boxes[:, 2] = (coord[:, :, 2].detach().exp() * anchor_w).view(-1)
#         pred_boxes[:, 3] = (coord[:, :, 3].detach().exp() * anchor_h).view(-1)
#         pred_boxes = pred_boxes.cpu()
#
#         # Get target values
#         coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls = self.build_targets(pred_boxes, target, height, width)
#         coord_mask = coord_mask.expand_as(tcoord)
#         tcls = tcls[cls_mask].view(-1).long()
#         cls_mask = cls_mask.view(-1, 1).repeat(1, self.num_classes)
#
#         if torch.cuda.is_available():
#             tcoord = tcoord.cuda()
#             tconf = tconf.cuda()
#             coord_mask = coord_mask.cuda()
#             conf_mask = conf_mask.cuda()
#             tcls = tcls.cuda()
#             cls_mask = cls_mask.cuda()
#
#         conf_mask = conf_mask.sqrt()
#         cls = cls[cls_mask].view(-1, self.num_classes)
#
#         # Compute losses
#         mse = nn.MSELoss(size_average=False)
#         ce = nn.CrossEntropyLoss(size_average=False)
#         self.loss_coord = self.coord_scale * mse(coord * coord_mask, tcoord * coord_mask) / batch_size
#         self.loss_conf = mse(conf * conf_mask, tconf * conf_mask) / batch_size
#         self.loss_cls = self.class_scale * 2 * ce(cls, tcls) / batch_size
#         self.loss_tot = self.loss_coord + self.loss_conf + self.loss_cls
#
#         return self.loss_tot, self.loss_coord, self.loss_conf, self.loss_cls
#
#     def build_targets(self, pred_boxes, ground_truth, height, width):
#         batch_size = len(ground_truth)
#
#         conf_mask = torch.ones(batch_size, self.num_anchors, height * width, requires_grad=False) * self.noobject_scale
#         coord_mask = torch.zeros(batch_size, self.num_anchors, 1, height * width, requires_grad=False)
#         cls_mask = torch.zeros(batch_size, self.num_anchors, height * width, requires_grad=False).byte()
#         tcoord = torch.zeros(batch_size, self.num_anchors, 4, height * width, requires_grad=False)
#         tconf = torch.zeros(batch_size, self.num_anchors, height * width, requires_grad=False)
#         tcls = torch.zeros(batch_size, self.num_anchors, height * width, requires_grad=False)
#
#         for b in range(batch_size):
#             if len(ground_truth[b]) == 0:
#                 continue
#
#             # Build up tensors
#             cur_pred_boxes = pred_boxes[
#                              b * (self.num_anchors * height * width):(b + 1) * (self.num_anchors * height * width)]
#             if self.anchor_step == 4:
#                 anchors = self.anchors.clone()
#                 anchors[:, :2] = 0
#             else:
#                 anchors = torch.cat([torch.zeros_like(self.anchors), self.anchors], 1)
#             gt = torch.zeros(len(ground_truth[b]), 4)
#             for i, anno in enumerate(ground_truth[b]):
#                 gt[i, 0] = (anno[0] + anno[2] / 2) / self.reduction
#                 gt[i, 1] = (anno[1] + anno[3] / 2) / self.reduction
#                 gt[i, 2] = anno[2] / self.reduction
#                 gt[i, 3] = anno[3] / self.reduction
#
#             # Set confidence mask of matching detections to 0
#             iou_gt_pred = bbox_ious(gt, cur_pred_boxes)
#             mask = (iou_gt_pred > self.thresh).sum(0) >= 1
#             conf_mask[b][mask.view_as(conf_mask[b])] = 0
#
#             # Find best anchor for each ground truth
#             gt_wh = gt.clone()
#             gt_wh[:, :2] = 0
#             iou_gt_anchors = bbox_ious(gt_wh, anchors)
#             _, best_anchors = iou_gt_anchors.max(1)
#
#             # Set masks and target values for each ground truth
#             for i, anno in enumerate(ground_truth[b]):
#                 gi = min(width - 1, max(0, int(gt[i, 0])))
#                 gj = min(height - 1, max(0, int(gt[i, 1])))
#                 best_n = best_anchors[i]
#                 iou = iou_gt_pred[i][best_n * height * width + gj * width + gi]
#                 coord_mask[b][best_n][0][gj * width + gi] = 1
#                 cls_mask[b][best_n][gj * width + gi] = 1
#                 conf_mask[b][best_n][gj * width + gi] = self.object_scale
#                 tcoord[b][best_n][0][gj * width + gi] = gt[i, 0] - gi
#                 tcoord[b][best_n][1][gj * width + gi] = gt[i, 1] - gj
#                 tcoord[b][best_n][2][gj * width + gi] = math.log(max(gt[i, 2], 1.0) / self.anchors[best_n, 0])
#                 tcoord[b][best_n][3][gj * width + gi] = math.log(max(gt[i, 3], 1.0) / self.anchors[best_n, 1])
#                 tconf[b][best_n][gj * width + gi] = iou
#                 tcls[b][best_n][gj * width + gi] = int(anno[4])
#
#         return coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls
#
#
# def bbox_ious(boxes1, boxes2):
#     b1x1, b1y1 = (boxes1[:, :2] - (boxes1[:, 2:4] / 2)).split(1, 1)
#     b1x2, b1y2 = (boxes1[:, :2] + (boxes1[:, 2:4] / 2)).split(1, 1)
#     b2x1, b2y1 = (boxes2[:, :2] - (boxes2[:, 2:4] / 2)).split(1, 1)
#     b2x2, b2y2 = (boxes2[:, :2] + (boxes2[:, 2:4] / 2)).split(1, 1)
#
#     dx = (b1x2.min(b2x2.t()) - b1x1.max(b2x1.t())).clamp(min=0)
#     dy = (b1y2.min(b2y2.t()) - b1y1.max(b2y1.t())).clamp(min=0)
#     intersections = dx * dy
#
#     areas1 = (b1x2 - b1x1) * (b1y2 - b1y1)
#     areas2 = (b2x2 - b2x1) * (b2y2 - b2y1)
#     unions = (areas1 + areas2.t()) - intersections
#
#     return intersections / unions


#######################################################################################################################


class YoloLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.lambda_obj = 1
        self.lambda_no_obj = 10
        self.lambda_box = 10
        self.lambda_cls = 1
        self.anchors = torch.Tensor([[1.73145, 1.3221],
                                     [4.00944, 3.19275],
                                     [8.09892, 5.05587],
                                     [4.84053, 9.47112],
                                     [10.0071, 11.2364]])
        self.anchor_boxes = self._generate_anchor_box(anchor_box_sizes=self.anchors,
                                                      out_size=(13, 13))

    def forward(self, predict, target):
        pred = predict.reshape(predict.shape[0], -1, 5 + self.num_classes)
        tar = target.reshape(target.shape[0], -1, 5 + self.num_classes)

        objs = tar[..., 4] == 1
        no_objs = tar[..., 4] == 0

        # Box Loss
        loss_bbox = self.lambda_box * self.mse_loss(pred[objs][..., 0:2], tar[objs][..., 0:2]) + \
                    self.rmse_loss(pred[objs][..., 2:4], tar[objs][..., 2:4])

        # Obj Loss
        loss_obj = self.lambda_obj * self.cross_entropy_loss(pred[objs][..., 4], tar[objs][..., 4])

        # No Obj Loss
        loss_no_obj = self.lambda_no_obj * self.cross_entropy_loss(pred[no_objs][..., 4], tar[no_objs][..., 4])

        # Class Loss
        loss_cls = self.lambda_cls * self.cross_entropy_loss(pred[objs][..., 5:], tar[objs][..., 5:])

        loss = loss_bbox + loss_obj + loss_no_obj + loss_cls

        # print(loss_bbox.detach().cpu().item(),
        #       loss_obj.detach().cpu().item(),
        #       loss_no_obj.detach().cpu().item(),
        #       loss_cls.detach().cpu().item())

        return loss, loss_bbox, loss_obj, loss_no_obj, loss_cls

    def mse_loss(self, predict, target):
        loss = torch.square(predict - target)

        return loss.mean()

    def rmse_loss(self, predict, target):
        pred = torch.sqrt(predict)
        target = torch.sqrt(target)
        loss = torch.square(pred - target)

        return loss.mean()

    def cross_entropy_loss(self, predict, target, mode='mean'):
        loss = -(target * torch.log2(predict + 1e-9) + (1 - target) * torch.log2(1 - predict + 1e-9))

        if mode == 'sum':
            return loss.sum()
        elif mode == 'mean':
            return loss.mean()

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

        num_anchor_boxes = int(anchor_boxes.shape[2] / 4)
        num_data_per_box = int(deltas.shape[2] / num_anchor_boxes)

        anchor_boxes = anchor_boxes.unsqueeze(0)

        for i in range(num_anchor_boxes):
            out[..., num_data_per_box * i: num_data_per_box * i + 2] = sigmoid(
                deltas[..., num_data_per_box * i:num_data_per_box * i + 2]
            ) + anchor_boxes[..., num_data_per_box * i:num_data_per_box * i + 2]
            out[..., num_data_per_box * i + 2:num_data_per_box * i + 4] = torch.exp(
                deltas[..., num_data_per_box * i + 2:num_data_per_box * i + 4]
            ) * anchor_boxes[..., num_data_per_box * i + 2:num_data_per_box * i + 4]
            out[..., num_data_per_box * i + 4] = sigmoid(deltas[..., num_data_per_box * i + 4])
            out[..., num_data_per_box * i + 5:num_data_per_box * (i + 1)] = softmax(
                deltas[..., num_data_per_box * i + 5:num_data_per_box * (i + 1)]
            )

        return out



def yolov2_custom_loss_1(predict, target, anchor_boxes, num_bbox_predict, num_classes, lambda_coord=5, lambda_noobj=.5):
    """
    :param predict: tensor, [batch, height, width, (cy, cx, h, w, p) * num bounding boxes]
    :param target: tensor, [batch, height, width, (cy, cx, h, w, p) * num bounding boxes]
    :param anchor_boxes: tensor, [height, width, (y, x, h, w)]
    :param num_bbox_predict: int
    :param num_classes: int
    :param lambda_coord: float
    :param lambda_noobj: float

    :return: float tensor, [float]
    """
    NUM_BATCH, H, W = predict.shape[:3]

    pred = predict.reshape(NUM_BATCH, -1, 5 + num_classes)  # [num batch, h * w * 5(num predict bbox), 5(num coords) + num classes)]
    tar = target.reshape(NUM_BATCH, -1, 5 + num_classes)  # [num batch, h * w * 5(num predict bbox), 5(num coords) + num classes)]
    anc = anchor_boxes.reshape(-1, 4)

    # obj_responsible_mask = torch.zeros(NUM_BATCH, H * W, 5).to(device)

    # for i in range(num_bbox_predict):
    #     obj_responsible_mask[:, :, :, i] = target[:, :, :, 4]

    # Get responsible masks
    pred_bboxes = pred[:, :, :4]
    pred_probs = pred[:, :, 4]

    tar_bboxes = tar[:, :, :4]
    tar_probs = tar[:, :, 4]  # [num batch, h * w * 5(num predict bbox)]

    # pred_y1 = pred_bboxes[:, :, 0] - .5 * pred_bboxes[:, :, 2]
    # pred_x1 = pred_bboxes[:, :, 1] - .5 * pred_bboxes[:, :, 3]
    # pred_y2 = pred_bboxes[:, :, 0] + pred_bboxes[:, :, 2] * anc[:, :, 2]
    # pred_x2 = pred_bboxes[:, :, 1] + pred_bboxes[:, :, 3] * anc[:, :, 3]
    pred_y = pred_bboxes[:, :, 0] + anc[:, 0]
    pred_x = pred_bboxes[:, :, 1] + anc[:, 1]
    pred_h = pred_bboxes[:, :, 2] * anc[:, 2]
    pred_w = pred_bboxes[:, :, 3] * anc[:, 3]

    pred_y1 = pred_y - .5 * pred_h
    pred_x1 = pred_x - .5 * pred_w
    pred_y2 = pred_y + pred_h
    pred_x2 = pred_x + pred_w

    pred_bboxes = torch.cat([pred_y1.unsqueeze(2), pred_x1.unsqueeze(2), pred_y2.unsqueeze(2), pred_x2.unsqueeze(2)], dim=2)

    tar_y = tar_bboxes[:, :, 0] + anc[:, 0]
    tar_x = tar_bboxes[:, :, 1] + anc[:, 1]
    tar_h = tar_bboxes[:, :, 2] * anc[:, 2]
    tar_w = tar_bboxes[:, :, 3] * anc[:, 3]

    # tar_y1 = tar_bboxes[:, :, 0] - .5 * tar_bboxes[:, :, 2]
    # tar_x1 = tar_bboxes[:, :, 1] - .5 * tar_bboxes[:, :, 3]
    # tar_y2 = tar_bboxes[:, :, 0] + tar_bboxes[:, :, 2] * anc[:, 2]
    # tar_x2 = tar_bboxes[:, :, 1] + tar_bboxes[:, :, 3] * anc[:, 3]

    tar_y1 = tar_y - .5 * tar_h
    tar_x1 = tar_x - .5 * tar_w
    tar_y2 = tar_y + tar_h
    tar_x2 = tar_x + tar_w

    tar_bboxes = torch.cat([tar_y1.unsqueeze(2), tar_x1.unsqueeze(2), tar_y2.unsqueeze(2), tar_x2.unsqueeze(2)], dim=2)

    # for idx1 in range(NUM_BATCH):
    #     for idx2 in range(13 * 13 * 5):
    #         if tar[idx1, idx2, 0] != .5:
    #             print(f'{idx1 + 1} {pred[idx1, idx2, :5].detach().cpu().numpy()}, {tar[idx1, idx2, :5].detach().cpu().numpy()}')

    ########## Original (start) ##########
    indices_valid = torch.where(tar_probs > 0)
    # pred_bboxes_valid = pred_bboxes[indices_valid].reshape(NUM_BATCH, -1, 4)
    # tar_bboxes_valid = tar_bboxes[indices_valid].reshape(NUM_BATCH, -1, 4)
    pred_bboxes_valid = pred_bboxes[indices_valid].reshape(-1, 4)
    tar_bboxes_valid = tar_bboxes[indices_valid].reshape(-1, 4)

    ious_valid = calculate_iou(pred_bboxes_valid, tar_bboxes_valid, dim=1).reshape(-1)

    ious = torch.zeros(NUM_BATCH, H * W * 5).to(device)  # [num batch, h * w * 5(num predict bbox)]
    ious[indices_valid] = ious_valid
    ious = ious.reshape(NUM_BATCH, H * W, 5)
    ########## Original (end) ##########
    ########## Changed (start) ##########
    ious = calculate_iou(pred_bboxes, tar_bboxes, dim=2).reshape(NUM_BATCH, -1, 5)
    ########## Changed (end) ##########


    # ious_temp = ious.reshape(NUM_BATCH, -1)
    # for idx1 in range(NUM_BATCH):
    #     for idx2 in range(13 * 13 * 5):
    #         if tar_probs[idx1, idx2].detach().cpu().numpy() > 0:
    #             print('[{}] {}'.format(idx1, ious_temp[idx1, idx2].detach().cpu().numpy()))

    indices_argmax_ious = torch.argmax(ious, dim=2)
    idx1 = []
    for i in range(indices_argmax_ious.shape[0]):
        idx1 += [i for _ in range(indices_argmax_ious.shape[1])]

    idx2 = []
    for i in range(indices_argmax_ious.shape[0]):
        idx2 += [j for j in range(indices_argmax_ious.shape[1])]

    idx3 = indices_argmax_ious.reshape(-1).squeeze()

    obj_responsible_mask = torch.zeros(NUM_BATCH, H * W, 5).to(device)  # [num batch, h * w, 5(num predict bbox)]
    obj_responsible_mask[idx1, idx2, idx3] = 1
    ########## Added (start) ########## - 2021.03.03
    obj_responsible_mask *= tar_probs.reshape(NUM_BATCH, -1, 5)
    ########## Added (end) ########## - 2021.03.03
    # obj_responsible_mask[indices_valid] = 1
    obj_responsible_mask = obj_responsible_mask.reshape(NUM_BATCH, -1, 5)

    # for i in range(NUM_BATCH):
    #     for j in range(13 * 13):
    #         if 1 in obj_responsible_mask[i, j]:
    #             print('[{}] ({}) {}'.format(i, j, obj_responsible_mask[i, j]))

    ########## Original (start) ########## - 2021.03.02
    # no_obj_responsible_mask = torch.zeros(NUM_BATCH, H * W, 5).to(device)
    # no_obj_responsible_mask[indices_argmax_ious[:-1]] = 1
    # no_obj_responsible_mask[indices_argmax_ious] = 0
    ########## Original (end) ########## - 2021.03.02
    ########## Changed (start) ########## - 2021.03.02
    no_obj_responsible_mask = 1 - obj_responsible_mask
    ########## Changed (end) ########## - 2021.03.02

    # Get coordinate loss(1)
    loss_coord = torch.square(pred[:, :, 0] - tar[:, :, 0]) + \
                 torch.square(pred[:, :, 1] - tar[:, :, 1]) + \
                 torch.square(torch.sqrt(pred[:, :, 2]) - torch.sqrt(tar[:, :, 2])) + \
                 torch.square(torch.sqrt(pred[:, :, 3]) - torch.sqrt(tar[:, :, 3]))
    loss_coord *= lambda_coord * obj_responsible_mask.reshape(NUM_BATCH, -1)

    # for i in range(pred.shape[1]):
    #     if tar[0, i, 4] == 1:
    #         print(pred[0, i, 4], tar[0, i, 4], ious.reshape(NUM_BATCH, -1)[0, i])

    # Get confidence loss(2)
    loss_confidence = obj_responsible_mask.reshape(NUM_BATCH, -1) * torch.square(pred[:, :, 4] - tar[:, :, 4] * ious.reshape(NUM_BATCH, -1)) + \
                      lambda_noobj * no_obj_responsible_mask.reshape(NUM_BATCH, -1) * \
                      torch.square(pred[:, :, 4] - tar[:, :, 4])

    # for idx1 in range(NUM_BATCH):
    #     for idx2 in range(13 * 13 * 5):
    #         if obj_responsible_mask.reshape(NUM_BATCH, -1)[idx1, idx2] == 1:
    #             print(f'PRED: {pred[idx1, idx2, 4]}, TAR: {tar[idx1, idx2, 4] * ious.reshape(NUM_BATCH, -1)[idx1, idx2]}')

    # ious_temp = ious.reshape(NUM_BATCH, -1)
    # obj_mask_temp = obj_responsible_mask.reshape(NUM_BATCH, -1)
    # no_obj_mask_temp = no_obj_responsible_mask.reshape(NUM_BATCH, -1)
    # for i in range(NUM_BATCH):
    #     for j in range(13 * 13 * 5):
    #         if obj_mask_temp[i, j] == 1:
    #             print('[{}] {:.5f}  {}  {:.5f}'.format(i + 1, pred[i, j, 4].detach().cpu().item(), tar[i, j, 4].detach().cpu().item(), ious_temp[i, j].item()))
            # if no_obj_mask_temp[i, j] == 1:
            #     print('{:.5f}  {} / {}  {}'.format(
            #         pred[i, j, 4].detach().cpu().item(), tar[i, j, 4].detach().cpu().item(), obj_mask_temp[i, j], no_obj_mask_temp[i, j]))


    # Get class loss(3)
    loss_class = torch.square(pred[:, :, 5:] - tar[:, :, 5:])
    ########## Original (start) ########## - 2021.03.02
    # loss_class = loss_class.reshape(NUM_BATCH, H * W, -1)
    ########## Original (end) ########## - 2021.03.02
    loss_class = torch.sum(loss_class, dim=2)
    ########## Original (start) ########## - 2021.03.02
    # loss_class *= responsible_mask.reshape(NUM_BATCH, -1)
    ########## Original (end) ########## - 2021.03.02
    ########## Changed (start) ########## - 2021.03.02
    loss_class *= obj_responsible_mask.reshape(NUM_BATCH, -1)
    ########## Changed (end) ########## - 2021.03.02

    # for i in range(NUM_BATCH):
    #     for j in range(13 * 13):
    #         for m in range(5):
    #             if loss_class.reshape(NUM_BATCH, -1, 5)[i, j, m] != 0:
    #                 print(i, j, m, loss_class.reshape(NUM_BATCH, -1, 5)[i, j, m])

    # Sum up all the losses
    loss_coord = loss_coord.sum() / NUM_BATCH
    loss_confidence = loss_confidence.sum() / NUM_BATCH
    loss_class = loss_class.sum() / NUM_BATCH
    loss = loss_coord + loss_confidence + loss_class

    # if loss.detach().cpu().item() > 1000:
    #     print('bbox : ', tar_bboxes_valid)
    #     print('probs : ', tar_probs)

    return loss, loss_coord, loss_confidence, loss_class


def yolov2_custom_loss_2(predict, target, anchor_boxes, num_bbox_predict, num_classes, lambda_coord=5, lambda_noobj=.5):
    """
    :param predict: tensor, [batch, height, width, (cy, cx, h, w, p) * num bounding boxes]
    :param target: tensor, [batch, height, width, (cy, cx, h, w, p) * num bounding boxes]
    :param num_bbox_predict: int
    :param num_classes: int
    :param lambda_coord: float
    :param lambda_noobj: float

    :return: float tensor, [float]
    """

    h, w = predict.shape[1:3]

    coord_loss = torch.zeros(1).to(device)
    confidence_loss = torch.zeros(1).to(device)
    class_loss = torch.zeros(1).to(device)

    n_batch = predict.shape[0]
    for b in range(n_batch):
        obj_responsible_mask = torch.zeros(h, w, num_bbox_predict).to(device)
        no_obj_responsible_mask = torch.zeros(h, w, num_bbox_predict).to(device)

        # Get responsible box masks
        for i in range(num_bbox_predict):
            obj_responsible_mask[:, :, i] = target[b, :, :, (5 + num_classes) * i + 4]
            no_obj_responsible_mask[:, :, i] = target[b, :, :, (5 + num_classes) * i + 4]

        for s1 in range(7):
            for s2 in range(7):
                if obj_responsible_mask[s1, s2, 0] == 1:
                    ious = torch.zeros(num_bbox_predict)

                    for n in range(num_bbox_predict):
                        box_temp = convert_box_from_hw_to_yx(predict[b, s1, s2, (5 + num_classes) * n:(5 + num_classes) * n + 4]).to(device)
                        gt = target[b, s1, s2, :4]
                        ious[n] = calculate_iou(box_temp, gt)

                    idx_max_iou = ious.argmax().item()

                    for n in range(num_bbox_predict):
                        if n != idx_max_iou:
                            obj_responsible_mask[s1, s2, n] = 0
                        else:
                            no_obj_responsible_mask[s1, s2, n] = 0

        responsible_mask = torch.zeros(h, w).to(device)
        for n in range(num_bbox_predict):
            responsible_mask += obj_responsible_mask[:, :, n]

        # Calculate losses
        coord_loss_batch = torch.zeros(1).to(device)
        confidence_loss_batch = torch.zeros(1).to(device)
        class_loss_batch = torch.zeros(1).to(device)

        for i in range(num_bbox_predict):
            # Coordinate loss
            coord_losses_temp = torch.square(predict[b, :, :, (5 + num_classes) * i] - target[b, :, :, (5 + num_classes) * i]) \
                                + torch.square(predict[b, :, :, (5 + num_classes) * i + 1] - target[b, :, :, (5 + num_classes) * i + 1]) \
                                + torch.square(torch.sqrt(predict[b, :, :, (5 + num_classes) * i + 2]) - torch.sqrt(target[b, :, :, (5 + num_classes) * i + 2])) \
                                + torch.square(torch.sqrt(predict[b, :, :, (5 + num_classes) * i + 3]) - torch.sqrt(target[b, :, :, (5 + num_classes) * i + 3]))
            coord_losses_temp *= obj_responsible_mask[:, :, i]
            coord_loss_batch += coord_losses_temp.sum()

            # Confidence loss
            confidence_losses_temp = torch.square(predict[b, :, :, (5 + num_classes) * i + 4] - target[b, :, :, (5 + num_classes) * i + 4])
            confidence_loss_batch += (confidence_losses_temp * obj_responsible_mask[:, :, i] \
                                     + lambda_noobj * confidence_losses_temp * no_obj_responsible_mask[:, :, i]).sum()

            # Class loss
            class_losses_temp = torch.square(predict[b, :, :, (5 + num_classes) * i + 5:(5 + num_classes) * (i + 1)] -
                                             target[b, :, :, (5 + num_classes) * i + 5:(5 + num_classes) * (i + 1)]).sum(dim=2)
            class_loss_batch += (responsible_mask * class_losses_temp).sum()

        coord_loss += coord_loss_batch
        confidence_loss += confidence_loss_batch
        class_loss += class_loss_batch

    loss = (coord_loss + confidence_loss + class_loss) / n_batch
    # print(coord_loss.detach().cpu().item(), confidence_loss.detach().cpu().item(), class_loss.detach().cpu().item())

    return loss, coord_loss / n_batch, confidence_loss / n_batch, class_loss / n_batch

