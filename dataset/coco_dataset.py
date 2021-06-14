import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from pycocotools.coco import COCO

from dataset.augment import rotate2d, horizontal_flip


class COCODataset(data.Dataset):
    def __init__(self, root, images_dir, annotation_path, image_size, is_categorical=False, transforms=None, instance_seg=False, rotate_angle=None, do_horizontal_flip=False):
        self.root = root
        self.images_dir = images_dir
        self.coco = COCO(annotation_path)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.is_categorical = is_categorical
        self.transforms = transforms
        self.instance_seg = instance_seg
        if isinstance(rotate_angle, int):
            self.rotate_angle = (-rotate_angle, rotate_angle)
        else:
            self.rotate_angle = rotate_angle
        self.do_horizontal_flip = do_horizontal_flip

        self.num_classes = 92  # Background included
        self.image_size = image_size

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

        if self.transforms is not None:
            img = self.transforms(img)
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
        iscrowd = torch.zeros((num_objs, ), dtype=torch.int64)

        if self.rotate_angle is not None:
            if np.random.rand() < .7:
                angle = np.random.randint(self.rotate_angle[0], self.rotate_angle[1])
                img, boxes = rotate2d(image=img, bounding_box=boxes, angle=angle)
        if self.do_horizontal_flip:
            if np.random.rand() < .5:
                img, boxes = horizontal_flip(image=img, bounding_box=boxes)

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

        return img, my_annotation

    def __len__(self):
        return len(self.ids)

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