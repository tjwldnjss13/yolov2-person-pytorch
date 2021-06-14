import numpy as np

from utils.util import calculate_ious
from utils.pytorch_util import calculate_iou


def generate_anchor(feature_size, anchor_stride):
    # Inputs:
    #    feature_size: (h, w)
    #    anchor_stride: Int
    # Outputs:
    #    anchors: numpy array [-1, (cy, cx)]

    h_f, w_f = feature_size
    xs_ctr = np.arange(anchor_stride, (w_f + 1) * anchor_stride, anchor_stride)
    ys_ctr = np.arange(anchor_stride, (h_f + 1) * anchor_stride, anchor_stride)

    anchors = np.zeros((len(xs_ctr) * len(ys_ctr), 2))

    c_i = 0
    for x in xs_ctr:
        for y in ys_ctr:
            anchors[c_i, 1] = x - anchor_stride // 2
            anchors[c_i, 0] = y - anchor_stride // 2
            c_i += 1

    print('Anchors generated')

    return anchors


def generate_anchor_box(box_samples, input_size, anchor_stride):
    # Inputs:
    #    box_samples: tensor, [# box, (y1, x1, y2, x2)]
    #    input_size: tuple, (height, width)
    #    anchor_stride: int

    h_in, w_in = input_size
    h_feat, w_feat = h_in // anchor_stride, w_in // anchor_stride
    anchors = generate_anchor((h_feat, w_feat), anchor_stride)
    anchor_boxes = torch.zeros((len(anchors) * len(box_samples)), 4)

    i_anc = 0
    for anc in anchors:
        y_anc, x_anc = anc
        for box in box_samples:
            h, w = box[2] - box[0], box[3] - box[1]
            anchor_boxes[i_anc, 0] = y_anc - .5 * h
            anchor_boxes[i_anc, 1] = x_anc - .5 * w
            anchor_boxes[i_anc, 2] = y_anc + .5 * h
            anchor_boxes[i_anc, 3] = x_anc + .5 * w

            i_anc += 1

    valid_mask_1 = (anchor_boxes[:, 0] >= 0)
    valid_mask_2 = (anchor_boxes[:, 1] >= 0)
    valid_mask_3 = (anchor_boxes[:, 2] <= input_size[0])
    valid_mask_4 = (anchor_boxes[:, 3] <= input_size[1])

    valid_mask = valid_mask_1 * valid_mask_2 * valid_mask_3 * valid_mask_4
    valid_mask = torch.as_tensor(valid_mask)

    print('Anchor boxes generated')

    return anchor_boxes, valid_mask


def generate_anchor_box_with_ratio_scale(ratios, scales, input_size, anchor_stride):
    # Inputs:
    #    ratios: list
    #    sclaes: list
    #    input_size: (height, width)
    #    anchor_stride: int
    # Outputs:
    #    anchor_boxes: tensor [the number of anchor boxes, (y1, x1, y2, x2)]
    #    valid_mask: tensor [the number of anchor boxes]

    # 50 = 800 // 16

    # ratios = [.5, 1, 2]
    # scales = [128, 256, 512]

    in_h, in_w = input_size

    feat_h, feat_w = in_h // anchor_stride, in_w // anchor_stride
    anchors = generate_anchor((feat_h, feat_w), anchor_stride)
    anchor_boxes = torch.zeros((len(anchors) * len(ratios) * len(scales), 4))

    anc_i = 0
    for anc in anchors:
        anc_y, anc_x = anc
        for r in ratios:
            for s in scales:
                # h = anchor_stride * s * np.sqrt(r)
                # w = anchor_stride * s * np.sqrt(1. / r)

                # if r < 1:
                #     h, w = s, s * (1. / r)
                # elif r > 1:
                #     h, w = s * (1. / r), s
                # else:
                #     h, w = s, s

                if r < 1:
                    h, w = np.sqrt(s ** 2 / r), s ** 2 / np.sqrt(s ** 2 / r)
                elif r > 1:
                    h, w = s ** 2 / np.sqrt(s ** 2 * r), np.sqrt(s ** 2 * r)
                else:
                    h, w = s, s

                anchor_boxes[anc_i, 0] = anc_y - .5 * h
                anchor_boxes[anc_i, 1] = anc_x - .5 * w
                anchor_boxes[anc_i, 2] = anc_y + .5 * h
                anchor_boxes[anc_i, 3] = anc_x + .5 * w

                # anchor_boxes[anc_i, 0] = anc_y
                # anchor_boxes[anc_i, 1] = anc_x
                # anchor_boxes[anc_i, 2] = h
                # anchor_boxes[anc_i, 3] = w

                anc_i += 1

    valid_mask_1 = (anchor_boxes[:, 0] >= 0)
    valid_mask_2 = (anchor_boxes[:, 1] >= 0)
    valid_mask_3 = (anchor_boxes[:, 2] <= input_size[0])
    valid_mask_4 = (anchor_boxes[:, 3] <= input_size[1])

    valid_mask = valid_mask_1 * valid_mask_2 * valid_mask_3 * valid_mask_4
    valid_mask = torch.as_tensor(valid_mask)

    # idx_valid = np.where((anchor_boxes[:, 0] >= 0) &
    #                      (anchor_boxes[:, 1] >= 0) &
    #                      (anchor_boxes[:, 2] <= in_h) &
    #                      (anchor_boxes[:, 3] <= in_w))[0]
    # anchor_boxes = anchor_boxes[idx_valid]

    print('Anchor boxes generated')

    return anchor_boxes, valid_mask


def generate_anchor_box_categorical(anchor_boxes, ground_truth):
    n_gt = ground_truth.shape[0]
    ious_anc_gt = calculate_ious(anchor_boxes, ground_truth)
    argmax_iou_anc_gt = np.argmax(ious_anc_gt, axis=1)

    anchor_boxes_cat = [[] for _ in range(n_gt)]
    for i, arg in enumerate(argmax_iou_anc_gt):
        anchor_boxes_cat[arg].append(anchor_boxes[i])

    # anchor_gts = ground_truth[argmax_iou_anc_gt]

    for i in range(n_gt):
        anchor_boxes_cat[i] = np.array(anchor_boxes_cat[i])
    # anchor_boxes_cat = np.array(anchor_boxes_cat)

    print('Categorical anchor boxes generated')

    return anchor_boxes_cat


def generate_anchor_label(anchor_boxes, ground_truth, pos_threshold, neg_threshold):
    ious_anc_gt = calculate_ious(anchor_boxes, ground_truth)

    pos_args_ious_anc_gt_1 = np.argmax(ious_anc_gt, axis=0)
    pos_args_ious_anc_gt_2 = np.where(ious_anc_gt >= pos_threshold)[0]
    pos_args_ious_anc_gt = np.append(pos_args_ious_anc_gt_1, pos_args_ious_anc_gt_2)
    pos_args_ious_anc_gt = np.array(list(set(pos_args_ious_anc_gt)))

    # anchor_labels = np.zeros(anchors.shape[0])
    anchor_labels = np.array([-1 for _ in range(anchor_boxes.shape[0])])
    anchor_labels[pos_args_ious_anc_gt] = 1

    non_pos_args_labels = np.where(anchor_labels != 1)[0]
    for i in non_pos_args_labels:
        neg_f = False
        for j in range(len(ground_truth)):
            if ious_anc_gt[i, j] >= neg_threshold:
                break
            neg_f = True
        if neg_f:
            anchor_labels[i] = 0

    # neg_args_ious_anc_gt = np.where(anchor_labels == -1)[0]

    print('Anchor labels generated')

    return anchor_labels


def generate_anchor_label_2dim(anchor_labels):
    anchor_labels2 = np.zeros((anchor_labels.shape[0], 2))
    train_args = np.where(anchor_labels != -1)
    anchor_labels2[train_args, anchor_labels[train_args]] = 1

    print('2-dim anchor labels generated')

    return anchor_labels2


def generate_anchor_ground_truth(anchor_boxes, ground_truth):
    ious_anc_gt = calculate_ious(anchor_boxes, ground_truth)
    argmax_iou_anc_gt = np.argmax(ious_anc_gt, axis=1)

    anchor_gts = ground_truth[argmax_iou_anc_gt]

    print('Anchor ground truth generated')

    return anchor_gts


def calculate_location_delta(bbox, anchor_box):
    # Inputs:
    #    bbox: tensor [# of bounding box, (y1, x1, y2, x2)]
    #    anchor_box: tensor [# of bounding box, (y1, x1, y2, x2)]
    # Outputs:
    #    locs: tensor [# of bounding box, (cy, cx, h, w)]

    assert bbox.shape == anchor_box.shape

    bbox_h, bbox_w = bbox[:, 2] - bbox[:, 0], bbox[:, 3] - bbox[:, 1]
    bbox_cy, bbox_cx = bbox[:, 0] + .5 * bbox_h, bbox[:, 1] + .5 * bbox_w

    anc_h, anc_w = anchor_box[:, 2] - anchor_box[:, 0], anchor_box[:, 3] - anchor_box[:, 1]
    anc_cy, anc_cx = anchor_box[:, 0] + .5 * anc_h, anchor_box[:, 1] + .5 * anc_w

    loc_h, loc_w = torch.log(bbox_h / anc_h), torch.log(bbox_w / anc_w)
    loc_cy, loc_cx = (bbox_cy - anc_cy) / anc_h, (bbox_cx - anc_cx) / anc_w

    locs = torch.zeros(bbox.shape)
    locs[:, 0], locs[:, 1], locs[:, 2], locs[:, 3] = loc_cy, loc_cx, loc_h, loc_w

    return locs


def non_maximum_suppression(bbox, score, threshold=.5):
    # Inputs:
    #    bbox: tensor [# batch, # bounding box, (y1, x1, y2, x2)]
    #    score: tensor [# batch, # bounding box]
    #    threshold: Float
    # Outputs:
    #    bbox_nms: tensor [# batch, # bounding box, (y1, x1, y2, x2)]
    #    score_nms: tensor [# batch, # bounding box]

    keep = torch.ones(score.shape)
    v, idx = score.sort(descending=True)
    bbox_base = bbox[idx[0]]

    for b in range(bbox.shape[0]):
        for i in range(1, len(idx)):
            bbox_temp = bbox[idx[i]]
            iou = calculate_iou(bbox_base, bbox_temp)
            if iou > threshold:
                keep[i] = 0

    bbox_nms = bbox * keep.reshape(len(keep), -1)
    score_nms = score * keep

    return bbox_nms, score_nms


def generate_rpn_targets(predict_bboxes, ground_truth, anchor_boxes, image_size, in_size, out_size, valid_mask):
    # Inputs:
    #    predict_bboxes: tensor [# bounding box, (y1, x1, y2, x2)]
    #    predict_scores: tensor [# bounding box, (no obj score, obj score)]
    #    anchor_boxes: tensor [output height, output width, 4 * 9]
    #                      or [# bounding box, (y1, x1, y2, x2)] (currently latter)
    #    ground_truth: tensor [# ground truth bounding box, (y1, x1, y2, x2)]
    #    image_size: (depth, image height, image width)
    #    in_size: (input height, input width)
    #    out_size: (output height, output width)
    # Outputs:
    #    pred_ds: tensor [-1, (dy, dx, dh, dw)]
    #    target_ds: tensor [-1, (dy, dx, dh, dw)]
    #    target_prob: tensor [-1, p] (p==1: positive, p==0: negative, p==-1: non-train)

    assert predict_bboxes.shape == anchor_boxes.shape
    assert len(predict_bboxes) == len(anchor_boxes)

    # Define variables
    num_bboxes = len(anchor_boxes)
    num_ground_truth = len(ground_truth)

    # Define tensors
    target_bboxes = torch.zeros(anchor_boxes.shape)
    target_prob = -1 * torch.ones(num_bboxes)

    # # Calculate prediction delta vectors (dy, dx, dh, dw)
    # pred_deltas = calculate_location_delta(predict_bboxes, anchor_boxes)

    # Make target bounding box tensor
    for i, bbox in enumerate(predict_bboxes):
        if valid_mask[i] == 0:
            continue

        pred_y1, pred_x1, pred_y2, pred_x2 = bbox

        ious_gt_anc = torch.zeros(3)
        for j, gt in enumerate(ground_truth):
            iou_temp = calculate_iou(bbox, gt)
            ious_gt_anc[j] = iou_temp

        # Assign maximum-iou ground truth box to target bounding box tensor
        idx_gt_iou_max = ious_gt_anc.argmax()
        target_bboxes[i] = ground_truth[idx_gt_iou_max]

        # Assign positive label
        if True in (ious_gt_anc > .7):
            target_prob[i] = 1

        # Assign negative label
        if (ious_gt_anc < .3).sum().item() == num_ground_truth:
            target_prob[i] = -1

    # Calculate target delta vectors (*dy, *dx, *dh, *dw)
    target_deltas = calculate_location_delta(target_bboxes, anchor_boxes)

    # return pred_deltas, target_deltas, target_prob

    return target_deltas, target_prob



import torch
import cv2 as cv
import matplotlib.pyplot as plt

# anc_boxes, valid_mask = generate_anchor_box([.5, 1, 2], [64, 128, 256], (224, 224), 16)
# idx_valid = torch.nonzero((valid_mask == 1), as_tuple=False)
# valid_anc_boxes = anc_boxes[idx_valid]
#
# img = np.zeros((416, 416, 3))
#
# for i in range(9 * 80, 9 * 80 + 9):
#     y1, x1, y2, x2 = anc_boxes[i]
#     y1, x1, y2, x2 = int(y1.item()), int(x1.item()), int(y2.item()), int(x2.item())
#     cx = anc_boxes[i, 3] - anc_boxes[i, 1]
#     cy = anc_boxes[i, 2] - anc_boxes[i, 0]
#     cx, cy = int(cx.item()), int(cy.item())
#     print(y1, x1, y2, x2)
#
#     # cv.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), thickness=2)
#
#     if i % 3 == 0:
#         cv.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
#     elif i % 3 == 1:
#         cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
#     else:
#         cv.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
#
#
# plt.imshow(img)
# plt.show()

# anc_boxes = torch.as_tensor(anc_boxes * valid_mask.unsqueeze(1))
# anc_boxes = anc_boxes.reshape(14, 14, 36)
# anc_boxes = anc_boxes.permute(2, 0, 1)

# dummy = torch.Tensor(14 * 14 * 36, 4)
# gt = torch.Tensor(2, 4)
#
# lin = anc_boxes.reshape(-1)
# a = lin.reshape(14, 14, 36)







    






























