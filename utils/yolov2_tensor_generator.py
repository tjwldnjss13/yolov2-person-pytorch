import torch

from utils.pytorch_util import calculate_iou

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_output_anchor_box_tensor(anchor_box_sizes, out_size):
    """
    Make anchor box the same shape as output's.
    :param anchor_box_sizes: tensor, [# anchor box, (height, width)]
    :param out_size: tuple or list, (height, width)

    :return tensor, [height, width, (cy, cx, h, w) * num bounding box]
    """

    out = torch.zeros(out_size[0], out_size[1], 4 * len(anchor_box_sizes)).to(device)
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


# anc_sizes = torch.Tensor([[12, 23], [24, 15]])
# outs = get_output_anchor_box_tensor(anc_sizes, (13, 13))
# print(outs)


def get_yolo_v2_output_tensor(deltas, anchor_boxes):
    """
    :param deltas: tensor, [height, width, ((dy, dx, dh, dw, p) + class scores) * num anchor boxes]
    :param anchor_boxes: tensor, [height, width, (cy, cx, h, w) * num anchor boxes]

    :return: tensor, [height, width, ((cy, cx, h, w, p) + class scores) * num anchor boxes]
    """

    out = torch.zeros(deltas.shape).to(device)
    sigmoid = torch.nn.Sigmoid()
    softmax = torch.nn.Softmax(dim=2)
    tanh = torch.nn.Tanh()
    relu = torch.nn.ReLU()

    num_anchor_boxes = int(anchor_boxes.shape[2] / 4)
    num_data_per_box = int(deltas.shape[2] / num_anchor_boxes)

    for i in range(num_anchor_boxes):
        # out[:, :, num_data_per_box * i: num_data_per_box * i + 2] = \
        #     sigmoid(deltas[:, :, num_data_per_box * i:num_data_per_box * i + 2]) + \
        #     anchor_boxes[:, :, 4 * i:4 * i + 2]
        # out[:, :, num_data_per_box * i:num_data_per_box * i + 2] = torch.exp(deltas[:, :, num_data_per_box * i + 2:num_data_per_box * i + 4]) * \
        #                              anchor_boxes[:, :, 4 * i + 2:4 * (i + 1)]
        # out[:, :, num_data_per_box * i + 4] = sigmoid(deltas[:, :, num_data_per_box * i + 4])
        # out[:, :, num_data_per_box * i + 5:num_data_per_box * (i + 1)] = \
        #     softmax(deltas[:, :, num_data_per_box * i + 5:num_data_per_box * (i + 1)])
        
        ########## Original (start) ########## - 2021.03.02
        out[:, :, num_data_per_box * i: num_data_per_box * i + 2] = \
            sigmoid(deltas[:, :, num_data_per_box * i:num_data_per_box * i + 2])
        ########## Original (end) ########## - 2021.03.02
        ########## Changed (start) ########## - 2021.03.02
        # out[:, :, num_data_per_box * i: num_data_per_box * i + 2] = \
        #     relu(deltas[:, :, num_data_per_box * i:num_data_per_box * i + 2])
        ########## Changed (end) ########## - 2021.03.02
        ########## Original (start) ########## - 2021.03.02
        out[:, :, num_data_per_box * i + 2:num_data_per_box * i + 4] = \
            torch.exp(deltas[:, :, num_data_per_box * i + 2:num_data_per_box * i + 4])
        ########## Original (end) ########## - 2021.03.02
        ########## Changed (start) ########## - 2021.03.02
        # out[:, :, num_data_per_box * i:num_data_per_box * i + 2] = \
        #     deltas[:, :, num_data_per_box * i + 2:num_data_per_box * i + 4]
        ########## Changed (end) ########## - 2021.03.02
        out[:, :, num_data_per_box * i + 4] = sigmoid(deltas[:, :, num_data_per_box * i + 4])
        out[:, :, num_data_per_box * i + 5:num_data_per_box * (i + 1)] = \
            softmax(deltas[:, :, num_data_per_box * i + 5:num_data_per_box * (i + 1)])


    return out


# def get_yolo_v2_target_tensor(ground_truth_boxes, labels, n_bbox_predict, n_class, in_size, out_size):
#     """
#     :param ground_truth_boxes: tensor, [num ground truth, (y1, x1, y2, x2)]
#     :param labels: tensor, [num bounding boxes, (p0, p1, ..., pn)]
#     :param n_bbox_predict: int
#     :param n_class: int
#     :param in_size: tuple or list, (height, width)
#     :param out_size: tuple or list, (height, width)
#
#     :return: tensor, [height of output, width of output, (cy, cx, h, w, p) * num bounding boxes]
#     """
#
#     bboxes = ground_truth_boxes
#
#     n_gt = len(bboxes)
#     in_h, in_w = in_size
#     out_h, out_w = out_size
#
#     ratio_y = out_h / in_h
#     ratio_x = out_w / in_w
#
#     target = torch.zeros((out_h, out_w, 5 + n_class))
#
#     for i in range(n_gt):
#         bbox = bboxes[i]
#         h, w = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])  # Height, width is relative to original image
#         y, x = (bbox[0] + .5 * h) * ratio_y, (bbox[1] + .5 * w) * ratio_x
#
#         h, w = h * ratio_y, w * ratio_x
#
#         y_cell_idx, x_cell_idx = int(y), int(x)
#         y_cell, x_cell = y - int(y), x - int(x)
#         label = labels[i]
#
#         target[y_cell_idx, x_cell_idx, 0] = x_cell
#         target[y_cell_idx, x_cell_idx, 1] = y_cell
#         target[y_cell_idx, x_cell_idx, 2] = w
#         target[y_cell_idx, x_cell_idx, 3] = h
#         target[y_cell_idx, x_cell_idx, 4] = 1
#
#         target[y_cell_idx, x_cell_idx, 5:] = label
#
#     return target


def get_yolo_v2_target_tensor(ground_truth_boxes, anchor_boxes, labels, n_bbox_predict, n_class, in_size, out_size):
    """
    :param ground_truth_boxes: tensor, [num ground truth, (y1, x1, y2, x2)]
    :param anchor_boxes: tensor, [height, width, (cy, cx, h, w) * num bounding boxes]
    :param labels: tensor, [num bounding boxes, (p0, p1, ..., pn)]
    :param n_bbox_predict: int
    :param n_class: int
    :param in_size: tuple or list, (height, width)
    :param out_size: tuple or list, (height, width)

    :return: tensor, [height of output, width of output, (cy, cx, h, w, p) * num bounding boxes]
    """

    gt_bboxes = ground_truth_boxes

    n_gt = len(gt_bboxes)
    in_h, in_w = in_size
    out_h, out_w = out_size

    ratio_y = out_h / in_h
    ratio_x = out_w / in_w

    target = torch.zeros((out_h, out_w, (5 + n_class) * n_bbox_predict))
    for b in range(n_bbox_predict):
        target[:, :, (5 + n_class) * b:(5 + n_class) * b + 2] = .5
        target[:, :, (5 + n_class) * b + 2:(5 + n_class) * b + 4] = 1

    for i in range(n_gt):
        gt = gt_bboxes[i]
        if len(gt) == 0:
            continue
        h_gt, w_gt = (gt[2] - gt[0]), (gt[3] - gt[1])  # Height, width is relative to original image
        y_gt, x_gt = (gt[0] + .5 * h_gt) * ratio_y, (gt[1] + .5 * w_gt) * ratio_x

        h_gt, w_gt = h_gt * ratio_y, w_gt * ratio_x

        y_idx, x_idx = int(y_gt), int(x_gt)
        y, x = y_gt - int(y_gt), x_gt - int(x_gt)
        label = labels[i]

        ########## Original (start) ########## - 2021.03.03
        # iou_gt_anchor_list = []
        # for j in range(n_bbox_predict):
        #     # target[y_idx, x_idx, (5 + n_class) * j] = x
        #     # target[y_idx, x_idx, (5 + n_class) * j + 1] = y
        #     # target[y_idx, x_idx, (5 + n_class) * j + 2] = w
        #     # target[y_idx, x_idx, (5 + n_class) * j + 3] = h
        #     # target[y_idx, x_idx, (5 + n_class) * j + 4] = 1
        #     # target[y_idx, x_idx, (5 + n_class) * j + 5:(5 + n_class) * (j + 1)] = label
        #
        #     cy_anc, cx_anc, h_anc, w_anc = anchor_boxes[y_idx, x_idx, 4 * j:4 * (j + 1)]
        #     y1_anc = cy_anc - .5 * h_anc
        #     x1_anc = cx_anc - .5 * w_anc
        #     y2_anc = y1_anc + h_anc
        #     x2_anc = x1_anc + w_anc
        #     anc = torch.Tensor([y1_anc, x1_anc, y2_anc, x2_anc])
        #
        #     # print(f'GT: {gt / 32}, ANC: {anc}')
        #
        #     iou = calculate_iou(gt / 32, anc)
        #     iou_gt_anchor_list.append(iou.item())
        #
        # anc_idx = iou_gt_anchor_list.index(max(iou_gt_anchor_list))
        #
        # h_anc, w_anc = anchor_boxes[y_idx, x_idx, 4 * anc_idx + 2:4 * (anc_idx + 1)]
        # h, w = h_gt / h_anc, w_gt / w_anc
        # # ########## Added ########## - 2021.03.02
        # # h, w = torch.log(h + 1e-20), torch.log(w + 1e-20)
        #
        # target[y_idx, x_idx, (5 + n_class) * anc_idx] = y
        # target[y_idx, x_idx, (5 + n_class) * anc_idx + 1] = x
        # target[y_idx, x_idx, (5 + n_class) * anc_idx + 2] = h
        # target[y_idx, x_idx, (5 + n_class) * anc_idx + 3] = w
        # # target[y_idx, x_idx, (5 + n_class) * anc_idx + 4] = iou_gt_anchor_list[anc_idx]
        # target[y_idx, x_idx, (5 + n_class) * anc_idx + 4] = 1
        # target[y_idx, x_idx, (5 + n_class) * anc_idx + 5:(5 + n_class) * (anc_idx + 1)] = label
        ########## Original (end) ########## - 2021.03.03

        ########## Changed (start) ########## - 2021.03.03
        for anc_idx in range(5):
            h_anc, w_anc = anchor_boxes[y_idx, x_idx, 4 * anc_idx + 2:4 * (anc_idx + 1)]
            h, w = h_gt / h_anc, w_gt / w_anc

            target[y_idx, x_idx, (5 + n_class) * anc_idx] = y
            target[y_idx, x_idx, (5 + n_class) * anc_idx + 1] = x
            target[y_idx, x_idx, (5 + n_class) * anc_idx + 2] = h
            target[y_idx, x_idx, (5 + n_class) * anc_idx + 3] = w
            # target[y_idx, x_idx, (5 + n_class) * anc_idx + 4] = iou_gt_anchor_list[anc_idx]
            target[y_idx, x_idx, (5 + n_class) * anc_idx + 4] = 1
            target[y_idx, x_idx, (5 + n_class) * anc_idx + 5:(5 + n_class) * (anc_idx + 1)] = label
        ########## Changed (end) ########## - 2021.03.03

        # print(iou_gt_anchor_list[anc_idx])

    return target
