import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from pycocotools.coco import COCO

from dataset.augment import *


class YoloCOCODataset(data.Dataset):
    def __init__(self, root, images_dir, annotation_path, image_size, is_categorical=False, transform=None, augmentation=None):
        self.root = root
        self.images_dir = images_dir
        self.coco = COCO(annotation_path)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.is_categorical = is_categorical
        self.transform = transform
        self.transform_mask = transforms.Compose([transforms.Resize(image_size, interpolation=Image.NEAREST)])
        self.aug = augmentation

        self.num_classes = 92  # Background included
        self.image_size = image_size

        self.anchors = torch.Tensor([[1.73145, 1.3221],
                                       [4.00944, 3.19275],
                                       [8.09892, 5.05587],
                                       [4.84053, 9.47112],
                                       [10.0071, 11.2364]])
        self.anchor_boxes = self._generate_anchor_box(anchor_box_sizes=self.anchors,
                                                      out_size=(13, 13))

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        ann = coco.loadAnns(ann_ids)
        img_dict = coco.loadImgs(img_id)[0]
        img_fn = img_dict['file_name']
        img = Image.open(os.path.join(self.images_dir, img_fn)).convert('RGB')
        height = int(img_dict['height'])
        width = int(img_dict['width'])
        ratio_h, ratio_w = self.image_size[0] / height, self.image_size[1] / width

        if self.transform is not None:
            img = self.transform(img)
        else:
            transform = transforms.Compose([transforms.ToTensor()])
            img = transform(img)

        num_objs = len(ann)

        areas = []
        boxes = []
        category_names = []
        labels = []
        labels_categorical = []

        for i in range(num_objs):
            x_min = ann[i]['bbox'][0] * ratio_w
            y_min = ann[i]['bbox'][1] * ratio_h
            x_max = x_min + ann[i]['bbox'][2] * ratio_w
            y_max = y_min + ann[i]['bbox'][3] * ratio_h
            boxes.append([y_min, x_min, y_max, x_max])
            areas.append(ann[i]['area'])

            category_id = ann[i]['category_id']
            labels.append(category_id)
            labels_categorical.append(self.to_categorical(category_id, self.num_classes))
            category_names.append(coco.loadCats(category_id)[0]['name'])

        if len(ann) > 0:
            masks = coco.annToMask(ann[0])
            for i in range(1, num_objs):
                masks = masks | coco.annToMask(ann[i])
        else:
            masks = []

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        labels_categorical = torch.as_tensor(labels_categorical, dtype=torch.int64)
        img_id = torch.tensor([img_id])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        if self.aug is not None:
            if isinstance(self.aug, list):
                for func in self.aug:
                    img, boxes = func(img, boxes)
            else:
                img, boxes = self.aug(img, boxes)

        yolo_target = self._generate_yolo_target(ground_truth_boxes=boxes,
                                                 anchor_boxes=self.anchor_boxes,
                                                 labels=labels_categorical,
                                                 n_bbox_predict=5,
                                                 n_class=self.num_classes,
                                                 in_size=(416, 416),
                                                 out_size=(13, 13))

        my_annotation = {}
        my_annotation['height'] = height
        my_annotation['width'] = width
        my_annotation['mask'] = masks
        my_annotation['bbox'] = boxes
        my_annotation['label'] = labels
        my_annotation['label_categorical'] = labels_categorical
        my_annotation['image_id'] = img_id
        my_annotation['area'] = areas
        my_annotation['iscrowd'] = iscrowd
        my_annotation['category_name'] = category_names
        my_annotation['file_name'] = img_fn

        return img, yolo_target, my_annotation

    def __len__(self):
        return len(self.ids)

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

    def _generate_yolo_target(self, ground_truth_boxes, anchor_boxes, labels, n_bbox_predict, n_class, in_size, out_size):
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

        for i in range(n_gt):
            gt = gt_bboxes[i]
            if len(gt) == 0:
                continue
            h_gt, w_gt = (gt[2] - gt[0]) * ratio_y, (gt[3] - gt[1]) * ratio_x
            y_gt, x_gt = (gt[0] + .5 * h_gt) * ratio_y, (gt[1] + .5 * w_gt) * ratio_x

            y_idx, x_idx = int(y_gt), int(x_gt)
            label = labels[i]

            for anc_idx in range(5):
                h_anc, w_anc = anchor_boxes[y_idx, x_idx, 4 * anc_idx + 2:4 * (anc_idx + 1)]

                target[y_idx, x_idx, (5 + n_class) * anc_idx] = y_gt
                target[y_idx, x_idx, (5 + n_class) * anc_idx + 1] = x_gt
                target[y_idx, x_idx, (5 + n_class) * anc_idx + 2] = h_gt
                target[y_idx, x_idx, (5 + n_class) * anc_idx + 3] = w_gt
                target[y_idx, x_idx, (5 + n_class) * anc_idx + 4] = 1
                target[y_idx, x_idx, (5 + n_class) * anc_idx + 5:(5 + n_class) * (anc_idx + 1)] = label

        return target

    @staticmethod
    def to_categorical(label, num_classes):
        label_list = []
        if isinstance(label, list):
            for l in label:
                label_base = [0 for _ in range(num_classes)]
                label_base[l] = 1
                label_list.append(label_base)
        else:
            label_base = [0 for _ in range(num_classes)]
            label_base[label] = 1
            label_list.append(label_base)

        return label_list

    @staticmethod
    def to_categorical_multi_label(label, num_classes):
        label_result = [0 for _ in range(num_classes)]
        if isinstance(label, list):
            label_result
            for l in label:
                label_result[l] = 1
        else:
            label_result[label] = 1

        return label_result

def custom_collate_fn(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    ann = [item[2] for item in batch]
    # print('data: ', data)
    # print('target: ', target)
    return [data, target, ann]


def test():
    root = 'D://DeepLearningData/COCOdataset2017/'
    dset_name = 'coco2017'
    train_img_dir = os.path.join(root, 'images', 'train')
    val_img_dir = os.path.join(root, 'images', 'val')
    train_ann_pth = os.path.join(root, 'annotations', 'instances_train2017.json')
    val_ann_pth = os.path.join(root, 'annotations', 'instances_val2017.json')
    transform_og = transforms.Compose([transforms.Resize((416, 416)),
                                       transforms.ToTensor()])

    aug = [horizontal_flip_augmentation, rotate2d_augmentation]

    dset = YoloCOCODataset(root, train_img_dir, train_ann_pth, (416, 416), transform=transform_og, augmentation=aug)
    img, _, ann = dset[10]
    bbox = ann['bbox'].numpy()
    cats = ann['category_name']

    print(cats)

    img = img.permute(1, 2, 0).numpy()
    for b in bbox:
        img = cv.rectangle(img.copy(), (int(b[1]), int(b[0])), (int(b[3]), int(b[2])), (0, 255, 0), 2)

    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    test()

































