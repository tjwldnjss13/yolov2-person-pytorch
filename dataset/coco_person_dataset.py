import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from pycocotools.coco import COCO

from dataset.augment import rotate2d, horizontal_flip


class COCOPersonDataset(data.Dataset):
    def __init__(self, root, image_size, for_train=True, year='2017', is_categorical=False, transform=None):
        super().__init__()
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

        self.imgs_pth, self.anns = self.load_person_data()
        del self.images_dir, self.coco, self.ids

        self.is_categorical = is_categorical
        self.transform = transform

        self.num_classes = 92  # Background include
        if isinstance(image_size, tuple):
            self.image_size = image_size
        elif isinstance(image_size, int):
            self.image_size = (image_size, image_size)

    def load_person_data(self):
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

    def __getitem__(self, idx):
        img = Image.open(self.imgs_pth[idx])
        ann = self.anns[idx]
        bbox = torch.Tensor(ann['bbox'])

        y1 = bbox[..., 1] * self.image_size[0] / ann['height']
        x1 = bbox[..., 0] * self.image_size[1] / ann['width']
        y2 = y1 + bbox[..., 3] * self.image_size[0] / ann['height']
        x2 = x1 + bbox[..., 2] * self.image_size[1] / ann['width']

        bbox = torch.cat([y1.unsqueeze(-1), x1.unsqueeze(-1), y2.unsqueeze(-1), x2.unsqueeze(-1)], dim=-1).type(torch.int)

        img = self.transform(img)

        return img, bbox

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
    root = 'C://DeepLearningData/COCO/'
    transform = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
    dset = COCODataset(root=root, image_size=448, for_train=False, year='2017', transform=transform)

    for i in range(len(dset)):
        img, bbox = dset[i]

        img = img.permute(1, 2, 0).numpy()
        bbox = bbox.numpy()
        for b in bbox:
            img = cv.rectangle(img.copy(), (b[1], b[0]), (b[3], b[2]), (0, 255, 0), 2)
        plt.imshow(img)
        plt.show()






























