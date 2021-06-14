import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from pycocotools.coco import COCO

from dataset.augment import rotate2d, horizontal_flip
from dataset.coco_person_dataset import *


class YoloCOCODataset(data.Dataset):
    def __init__(self, root, image_size, for_train=True, year='2017', transform=None, augmentation=None):
        self.root = root
        if for_train:
            self.images_dir = os.path.join(self.root, 'images', 'train' + year)
            self.coco = COCO(os.path.join(self.root, 'annotations', 'instances_train' + year + '.json'))
        else:
            self.images_dir = os.path.join(self.root, 'images', 'val' + year)
            self.coco = COCO(os.path.join(self.root, 'annotations', 'instances_val' + year + '.json'))
        # self.images_dir = images_dir
        # self.coco = COCO(annotation_path)
        self.ids = list(sorted(self.coco.imgs.keys()))

        self.imgs_pth, self.anns = self._load_person_data()
        del self.images_dir, self.coco, self.ids

        self.transform = transform
        self.augmentation = augmentation

        if isinstance(image_size, tuple):
            self.image_size = image_size
        elif isinstance(image_size, int):
            self.image_size = (image_size, image_size)

        self.num_classes = 2
        self.anchors = torch.Tensor([[1.73145, 1.3221],
                                     [4.00944, 3.19275],
                                     [8.09892, 5.05587],
                                     [4.84053, 9.47112],
                                     [10.0071, 11.2364]])
        self.anchor_boxes = self._generate_anchor_box(anchor_box_sizes=self.anchors,
                                                      out_size=(13, 13))

    def _load_person_data(self):
        imgs = []
        anns_custom = []
        for i, id in enumerate(self.ids):
            bbox = []
            ann_custom = {}
            img_dict = self.coco.loadImgs(id)[0]
            img_fn = img_dict['file_name']
            anns = self.coco.loadAnns(self.coco.getAnnIds(id))
            for ann in anns:
                if ann['category_id'] == 1:
                    bbox.append(ann['bbox'])
            if len(bbox) == 0:
                continue
            imgs.append(os.path.join(self.images_dir, img_fn))
            ann_custom['bbox'] = bbox
            ann_custom['height'] = img_dict['height']
            ann_custom['width'] = img_dict['width']
            anns_custom.append(ann_custom)

        return imgs, anns_custom

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

    def _generate_yolo_target(self, ground_truth_boxes, anchor_boxes, n_bbox_predict, n_class, in_size, out_size):
        """
        :param ground_truth_boxes: tensor, [num ground truth, (y1, x1, y2, x2)]
        :param anchor_boxes: tensor, [height, width, (cy, cx, h, w) * num bounding boxes]
        :param n_bbox_predict: int
        :param n_class: int
        :param in_size: tuple or list, (height, width)
        :param out_size: tuple or list, (height, width)

        :return: tensor, [height of output, width of output, (cy, cx, h, w, p) * num bounding boxes]
        """

        def sigmoid_inverse(x):
            return torch.log(x / (1 - x) + 1e-9)

        gt_bboxes = ground_truth_boxes

        n_gt = len(gt_bboxes)
        in_h, in_w = in_size
        out_h, out_w = out_size

        ratio_h = in_h / out_h
        ratio_w = in_w / out_w

        target = torch.zeros((out_h, out_w, (5 + n_class) * n_bbox_predict))
        target[..., (5 - 1) + (n_class - 1):-1:5 + n_class] = 1

        for i in range(n_gt):
            gt = gt_bboxes[i]
            if len(gt) == 0:
                continue
            h_gt, w_gt = (gt[2] - gt[0]), (gt[3] - gt[1])
            y_gt, x_gt = (gt[0] + .5 * h_gt), (gt[1] + .5 * w_gt)

            y_idx, x_idx = int(y_gt), int(x_gt)

            for anc_idx in range(5):
                # h_anc, w_anc = anchor_boxes[y_idx, x_idx, 4 * anc_idx + 2:4 * (anc_idx + 1)]

                target[y_idx, x_idx, (5 + n_class) * anc_idx] = y_gt
                target[y_idx, x_idx, (5 + n_class) * anc_idx + 1] = x_gt
                target[y_idx, x_idx, (5 + n_class) * anc_idx + 2] = h_gt
                target[y_idx, x_idx, (5 + n_class) * anc_idx + 3] = w_gt
                target[y_idx, x_idx, (5 + n_class) * anc_idx + 4] = 1
                target[y_idx, x_idx, (5 + n_class) * anc_idx + 5] = 0
                target[y_idx, x_idx, (5 + n_class) * anc_idx + 6] = 1
                # target[y_idx, x_idx, (5 + n_class) * anc_idx] = sigmoid_inverse(y_gt)
                # target[y_idx, x_idx, (5 + n_class) * anc_idx + 1] = sigmoid_inverse(x_gt)
                # target[y_idx, x_idx, (5 + n_class) * anc_idx + 2] = torch.log(h_gt / h_anc + 1e-9)
                # target[y_idx, x_idx, (5 + n_class) * anc_idx + 3] = torch.log(w_gt / w_anc + 1e-9)
                # target[y_idx, x_idx, (5 + n_class) * anc_idx + 4] = 1
                # target[y_idx, x_idx, (5 + n_class) * anc_idx + 6] = 1

        return target

    def __getitem__(self, idx):
        img = Image.open(self.imgs_pth[idx]).convert('RGB')
        ann = self.anns[idx]
        bbox = torch.Tensor(ann['bbox'])

        y1 = bbox[..., 1] * self.image_size[0] / ann['height']
        x1 = bbox[..., 0] * self.image_size[1] / ann['width']
        y2 = y1 + bbox[..., 3] * self.image_size[0] / ann['height']
        x2 = x1 + bbox[..., 2] * self.image_size[1] / ann['width']

        bbox = torch.cat([y1.unsqueeze(-1), x1.unsqueeze(-1), y2.unsqueeze(-1), x2.unsqueeze(-1)], dim=-1).type(torch.int)
        img = self.transform(img)

        if self.augmentation is not None:
            for aug in self.augmentation:
                img, bbox = aug(img, bbox)

        # import matplotlib.pyplot as plt
        # import cv2 as cv
        # img_np = img.permute(1, 2, 0).numpy()
        # for b in bbox:
        #     img_np = cv.rectangle(img_np.copy(), (b[1], b[0]), (b[3], b[2]), (0, 255, 0), 2)
        # plt.imshow(img_np)
        # plt.show()

        yolo_target = self._generate_yolo_target(ground_truth_boxes=bbox * (13 / 416),
                                                 anchor_boxes=self.anchor_boxes,
                                                 n_bbox_predict=5,
                                                 n_class=self.num_classes,
                                                 in_size=(416, 416),
                                                 out_size=(13, 13))

        # import cv2 as cv
        # import matplotlib.pyplot as plt
        # img_np = img.permute(1, 2, 0).numpy()
        # for b in bbox:
        #     img_np = cv.rectangle(img_np.copy(), (b[1], b[0]), (b[3], b[2]), (0, 255, 0), 2)
        # plt.imshow(img_np)
        # plt.show()

        return img, yolo_target

    def __len__(self):
        return len(self.imgs_pth)

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
    # print('data: ', data)
    # print('target: ', target)
    return [data, target]


if __name__ == '__main__':
    import numpy as np
    import cv2 as cv
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    from utils.pytorch_util import *
    root = 'D://DeepLearningData/COCO/'
    transform = transforms.Compose([transforms.Resize((416, 416)), transforms.ToTensor()])
    dset = YoloCOCODataset(root=root,
                           image_size=(416, 416),
                           for_train=False,
                           transform=transform)
    img, tar = dset[0]
































