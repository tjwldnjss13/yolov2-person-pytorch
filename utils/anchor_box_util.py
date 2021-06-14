import numpy as np
import torch
import random

from utils.pytorch_util import calculate_iou
from dataset.voc_dataset import VOCDataset


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


def distance_metric(bbox1, bbox2):
    return 1 - calculate_iou(bbox1, bbox2)


def k_means_cluster_anchor_box(bbox, K):
    # Inputs:
    #    bbox: tensor, [# bounding box, (y1, x1, y2, x2)]
    #    K: int, # centroid
    # Outputs:
    #    centroids: tensor, [K, (y1, x1, y2, x2)]
    #    clusters: ndarray, [# bounding box]

    print('Start K-means clustering!')
    # centroids = torch.Tensor([[130.2030,  84.8202, 331.2180, 259.0452],
    #     [137.4449, 204.0276, 300.2263, 356.9088],
    #     [369.0522, 250.4970, 433.3239, 328.3164],
    #     [270.1581, 114.8591, 398.2650, 309.8760],
    #     [ 50.1476, 135.2850, 139.3589, 302.8188]])

    centroids = torch.zeros(1, 4)
    clusters = -1 * torch.ones(bbox.shape[0])
    changed = True
    max_iter = 100

    print('Selecting initial centroids...')
    idx_init_cent = random.randint(0, bbox.shape[0] - 1)
    centroids[0] = bbox[idx_init_cent]

    for i in range(1, K):
        max_dist = -1
        arg_max_dist = -1
        for b, box in enumerate(bbox):
            if box in centroids:
                continue
            # Distance metric is IoU, So maximum distance would not exceed 1.
            nearest_dist = 2
            for cent in centroids:
                dist_temp = distance_metric(box, cent)
                if dist_temp < nearest_dist:
                    nearest_dist = dist_temp
            if nearest_dist > max_dist:
                max_dist = nearest_dist
                arg_max_dist = b
        # centroids = np.concatenate([centroids, [bbox[arg_max_dist]]], axis=0)
        centroids = torch.cat([centroids, bbox[arg_max_dist].unsqueeze(0)], dim=0)

    print('[Centroids]')
    print(centroids)

    print('Clustering...')
    _iter = 0
    while changed and _iter < max_iter:
        print('[{}] '.format(_iter), end='')
        print('[', end='')
        for j in range(K):
            print('[{}, {}, {}, {}]'.format(int(centroids[j, 0].item()), int(centroids[j, 1].item()),
                                            int(centroids[j, 2].item()), int(centroids[j, 3].item())), end='')
            if j < K - 1:
                print(', ', end='')
        print(']')

        _iter += 1
        changed = False
        for b, box in enumerate(bbox):
            min_dist = -1
            arg_min_dist = -1
            for c, cent in enumerate(centroids):
                dist = calculate_iou(box, cent)
                if min_dist == -1 or dist < min_dist:
                    min_dist = dist
                    arg_min_dist = c
            if clusters[b] != arg_min_dist:
                clusters[b] = arg_min_dist
                changed = True

        if changed:
            for i in range(K):
                args = torch.where(clusters == i)
                bboxes_cluster = bbox[args]
                if len(bboxes_cluster) == 0:
                    continue
                x1, y1, x2, y2 = torch.mean(bboxes_cluster, dim=0)
                # new_cent = np.array(torch.Tensor([y1, x1, y2, x2]))
                new_cent = torch.Tensor([y1, x1, y2, x2])
                centroids[i] = new_cent

    print('K-means clustering done!')

    return centroids, clusters


def get_anchor_boxes_voc(voc_dataset, num_anchor_boxes):
    # Inputs:
    #    voc_dataset: custom VOC dataset object
    #    num_anchor_boxes: int, # anchor box

    bbox_list = voc_dataset.get_bounding_box_list()

    anchor_boxes, _ = k_means_cluster_anchor_box(bbox_list, num_anchor_boxes)

    return anchor_boxes



# root = 'C://DeepLearningData/VOC2012/'
# dset = VOCDataset(root, (1, 1))
# anchor_boxes = get_anchor_boxes_voc(dset, 5)
# print(anchor_boxes)

































