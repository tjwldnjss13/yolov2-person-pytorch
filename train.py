import os
import copy
import cv2 as cv
import numpy as np
import time
import argparse
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, ConcatDataset, Subset
from PIL import Image

from utils.util import time_calculator
from utils.pytorch_util import make_batch
from utils.yolov2_tensor_generator import get_output_anchor_box_tensor, get_yolo_v2_output_tensor, get_yolo_v2_target_tensor
# from dataset.coco_dataset import COCODataset, custom_collate_fn
# from dataset.voc_dataset import VOCDataset, custom_collate_fn
from dataset.yolo_dataset import *
from dataset.augment import *
from models.yolov2_model import YOLOV2Mobile
from loss import *
from early_stopping import EarlyStopping


def get_coco_dataset(root):
    from dataset.coco_dataset import COCODataset, custom_collate_fn
    dset_name = 'coco2017'
    train_img_dir = os.path.join(root, 'images', 'train')
    val_img_dir = os.path.join(root, 'images', 'val')
    train_ann_pth = os.path.join(root, 'annotations', 'instances_train2017.json')
    val_ann_pth = os.path.join(root, 'annotations', 'instances_val2017.json')
    transform_og = transforms.Compose([transforms.Resize((416, 416)),
                                       transforms.ToTensor()])
    transform_noise = transforms.Compose([transforms.Resize((416, 416)),
                                          transforms.ToTensor(),
                                          RandomGaussianNoise(mean=0, std=.2, probability=.7),
                                          transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

    train_dset = YoloCOCODataset(root=root,
                                 images_dir=train_img_dir,
                                 annotation_path=train_ann_pth,
                                 image_size=(416, 416),
                                 is_categorical=True,
                                 transform=transform_og)
    train_dset_flip = YoloCOCODataset(root=root,
                                     images_dir=train_img_dir,
                                     annotation_path=train_ann_pth,
                                     image_size=(416, 416),
                                     is_categorical=True,
                                     transform=transform_og,
                                     augmentation=horizontal_flip_augmentation)
    train_dset_rotate = YoloCOCODataset(root=root,
                                         images_dir=train_img_dir,
                                         annotation_path=train_ann_pth,
                                         image_size=(416, 416),
                                         is_categorical=True,
                                         transform=transform_og,
                                         augmentation=rotate2d_augmentation)
    train_dset_flip_rotate = YoloCOCODataset(root=root,
                                              images_dir=train_img_dir,
                                              annotation_path=train_ann_pth,
                                              image_size=(416, 416),
                                              is_categorical=True,
                                              transform=transform_og,
                                              augmentation=[horizontal_flip_augmentation, rotate2d_augmentation])

    # train_dset = COCODataset(root=root, images_dir=train_img_dir, annotation_path=train_ann_pth, image_size=(416, 416),
    #                          is_categorical=True, transforms=transform_noise, do_horizontal_flip=True)
    # train_dset_noise = COCODataset(root=root, images_dir=train_img_dir, annotation_path=train_ann_pth, image_size=(416, 416),
    #                                is_categorical=True, transforms=transform_noise, do_horizontal_flip=True)
    # train_dset_rotate = COCODataset(root=root, images_dir=train_img_dir, annotation_path=train_ann_pth, image_size=(416, 416),
    #                                 is_categorical=True, transforms=transform_og, rotate_angle=(-30, 30), do_horizontal_flip=True)

    num_classes = train_dset.num_classes

    train_dset = ConcatDataset([train_dset, train_dset_flip, train_dset_rotate, train_dset_flip_rotate])
    # val_dset = COCODataset(root=root, images_dir=val_img_dir, annotation_path=val_ann_pth, image_size=(416, 416),
    #                        is_categorical=True, transforms=transform_og)
    val_dset = YoloCOCODataset(root=root,
                               images_dir=val_img_dir,
                               annotation_path=val_ann_pth,
                               image_size=(416, 416),
                               is_categorical=True,
                               transform=transform_og)

    collate_fn = custom_collate_fn

    return dset_name, train_dset, val_dset, collate_fn


def get_voc_dataset(root):
    from dataset.voc_dataset import VOCDataset, custom_collate_fn
    dset_name = 'voc2012'

    transform_og = transforms.Compose([transforms.Resize((416, 416)),
                                       transforms.ToTensor()])
    transform_norm = transforms.Compose([transforms.Resize((416, 416)),
                                         transforms.ToTensor(),
                                         transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    transform_noise = transforms.Compose([transforms.Resize((416, 416)),
                                          transforms.ToTensor(),
                                          RandomGaussianNoise(mean=0, std=.2)])
    transform_norm_noise = transforms.Compose([transforms.Resize((416, 416)),
                                               transforms.ToTensor(),
                                               transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
                                               RandomGaussianNoise(mean=0, std=.2)])
    transform_rotate = transforms.Compose([transforms.Resize((416, 416)),
                                            transforms.RandomRotation((-60, 60)),
                                            transforms.ToTensor(),
                                            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    transform_vflip = transforms.Compose([transforms.Resize((416, 416)),
                                                   transforms.RandomVerticalFlip(1),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    transform_hflip = transforms.Compose([transforms.Resize((416, 416)),
                                                     transforms.RandomHorizontalFlip(1),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

    dset_og = VOCDataset(root, img_size=(416, 416), transforms=transform_og, is_categorical=True)
    dset_norm = VOCDataset(root, img_size=(416, 416), transforms=transform_norm, is_categorical=True)
    dset_noise = VOCDataset(root, img_size=(416, 416), transforms=transform_noise, is_categorical=True)
    dset_norm_noise = VOCDataset(root, img_size=(416, 416), transforms=transform_norm_noise, is_categorical=True)
    # dset_rotate = VOCDataset(root, img_size=(416, 416), transforms=transform_rotate, is_categorical=True)
    # dset_vflip = VOCDataset(root, img_size=(416, 416), transforms=transform_vflip, is_categorical=True)
    # dset_hflip = VOCDataset(root, img_size=(416, 416), transforms=transform_hflip, is_categorical=True)

    num_classes = dset_og.num_classes

    n_data = len(dset_og)
    n_train_data = int(n_data * .7)
    indices = list(range(n_data))

    np.random.shuffle(indices)
    train_idx, val_idx = indices[:n_train_data], indices[n_train_data:]
    train_dset_og = Subset(dset_og, indices=train_idx)
    train_dset_norm = Subset(dset_norm, indices=train_idx)
    train_dset_noise = Subset(dset_noise, indices=train_idx)
    train_dset_norm_noise = Subset(dset_norm_noise, indices=train_idx)
    # train_dset_rotate = Subset(dset_rotate, indices=train_idx)
    # train_dset_vflip = Subset(dset_vflip, indices=train_idx)
    # train_dset_hflip = Subset(dset_hflip, indices=train_idx)

    # train_dset = ConcatDataset([dset_og, dset_norm, dset_noise, dset_norm_noise])
    train_dset = train_dset_og
    val_dset = Subset(dset_og, indices=val_idx)

    collate_fn = custom_collate_fn

    return dset_name, train_dset, val_dset, collate_fn


def update_learning_rate(optimizer, current_epoch):
    for g in optimizer.param_groups:
        if current_epoch < 30:
            pass
        elif current_epoch < 60:
            g['lr'] *= .1
        else:
            g['lr'] *= .1


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Define hyper parameters, parsers
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, required=False, default=16)
    parser.add_argument('--lr', type=float, required=False, default=.0001)
    parser.add_argument('--weight_decay', type=float, required=False, default=.0005)
    parser.add_argument('--momentum', type=float, required=False, default=.9)
    parser.add_argument('--num_epochs', type=int, required=False, default=50)

    args = parser.parse_args()

    batch_size = args.batch_size
    learning_rate = args.lr
    weight_decay = args.weight_decay
    momentum = args.momentum
    num_epochs = args.num_epochs

    model_save_term = 1

    # Generate COCO dataset
    root = 'C://DeepLearningData/COCOdataset2017/'
    dset_name, train_dset, val_dset, collate_fn = get_coco_dataset(root)
    num_classes = 92

    # Generate VOC dataset
    # root = 'D://DeepLearningData/VOC2012'
    # dset_name, train_dset, val_dset, collate_fn = get_voc_dataset(root)
    # num_classes = 20

    # Generate data loaders
    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=custom_collate_fn)

    # Load model
    model_name = 'yolov2_darknet19'
    model = YOLOV2Mobile(in_size=(416, 416), num_classes=num_classes).to(device)
    state_dict_pth = None
    state_dict_pth = 'pretrained models/yolov2_darknet19_coco2017_26epoch_0.0001lr_5.73236loss_3.09900losscoord_1.57544lossobj_0.96959lossnoobj_0.08833losscls.pth'
    if state_dict_pth is not None:
        model.load_state_dict(torch.load(state_dict_pth), strict=False)

    # Define optimizer, loss function
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = optim.SGD(params=model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    loss_func = YoloLoss()

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=.9)

    num_iter = 0
    train_loss_list = []
    train_loss_coord_list = []
    train_loss_obj_list = []
    train_loss_no_obj_list = []
    train_loss_class_list = []
    val_loss_list = []
    val_loss_coord_list = []
    val_loss_obj_list = []
    val_loss_no_obj_list = []
    val_loss_class_list = []
    model.train()

    t_start = time.time()
    for e in range(num_epochs):
        num_batches = 0
        num_datas = 0
        train_loss = 0
        train_loss_coord = 0
        train_loss_obj = 0
        train_loss_no_obj = 0
        train_loss_class = 0

        # update_learning_rate(optimizer, e + 1)

        t_train_start = time.time()
        for i, (imgs, targets, anns) in enumerate(train_loader):
            t_batch_start = time.time()
            num_batches += 1
            num_datas += len(imgs)
            num_iter += 1
            print('[{}/{}] '.format(e + 1, num_epochs), end='')
            print(f'({num_iter}) ', end='')
            print('{}/{} '.format(num_datas, len(train_dset)), end='')

            x = make_batch(imgs).to(device)
            y = make_batch(targets).to(device)

            predict = model(x)

            optimizer.zero_grad()
            loss, loss_coord, loss_obj, loss_no_obj, loss_class = loss_func(predict=predict, target=y)


            if torch.isnan(loss).sum() > 0:
                print('NaN appears.')
                for ann in anns:
                    print(len(ann['bbox']))
                exit(0)

            loss.backward()
            optimizer.step()

            train_loss += loss.detach().cpu().item()
            train_loss_coord += loss_coord.detach().cpu().item()
            train_loss_obj += loss_obj.detach().cpu().item()
            train_loss_no_obj += loss_no_obj.detach().cpu().item()
            train_loss_class += loss_class.detach().cpu().item()

            t_batch_end = time.time()

            H, M, S = time_calculator(t_batch_end - t_start)

            print('<loss> {: <8.5f}  <loss_coord> {: <8.5f}  <loss_obj> {: <8.5f}  <loss_no_obj> {: <8.5f}  <loss_class> {: <8.5f}  '.format(
                loss.detach().cpu().item(), loss_coord.detach().cpu().item(),
                loss_obj.detach().cpu().item(), loss_no_obj.detach().cpu().item(), loss_class.detach().cpu().item()), end='')
            print('<loss_avg> {: <8.5f}  <loss_coord_avg> {: <8.5f}  <loss_obj_avg> {: <8.5f}  <loss_no_obj_avg> {: <8.5f}  <loss_class_avg> {: <8.5f}  '.format(
                train_loss / num_batches, train_loss_coord / num_batches, train_loss_obj / num_batches, train_loss_no_obj / num_batches, train_loss_class / num_batches
            ), end='')
            print('<time> {:02d}:{:02d}:{:02d}'.format(int(H), int(M), int(S)))

            # if num_iter > 0 and num_iter % 5000 == 0:
            #     save_dir = './saved models/'
            #     if not os.path.exists(save_dir):
            #         os.mkdir(save_dir)
            #     save_pth = 'saved models/ckp_{}_{}_{}epoch_{}iter_{}lr.pth'.format(
            #         model_name, dset_name, e + 1, num_iter, learning_rate)
            #     torch.save(model.state_dict(), save_pth)

            del y, predict, loss

        train_loss /= num_batches
        train_loss_coord /= num_batches
        train_loss_obj /= num_batches
        train_loss_no_obj /= num_batches
        train_loss_class /= num_batches

        train_loss_list.append(train_loss)
        train_loss_coord_list.append(train_loss_coord)
        train_loss_obj_list.append(train_loss_obj)
        train_loss_no_obj_list.append(train_loss_no_obj)
        train_loss_class_list.append(train_loss_class)

        t_train_end = time.time()
        H, M, S = time_calculator(t_train_end - t_train_start)

        print('        <train_loss> {: <8.5f}  <train_loss_coord> {: <8.5f}  <train_loss_obj> {: <8.5f}  <train_loss_no_obj> {: <8.5f}  <train_loss_class> {: <8.5f}  '.format(
            train_loss_list[-1], train_loss_coord_list[-1], train_loss_obj_list[-1], train_loss_no_obj_list[-1], train_loss_class_list[-1]), end='')
        print('<time> {:02d}:{:02d}:{:02d} '.format(int(H), int(M), int(S)))

        with torch.no_grad():
            model.eval()
            val_loss = 0
            val_loss_coord = 0
            val_loss_obj = 0
            val_loss_no_obj = 0
            val_loss_class = 0
            num_batches = 0

            for i, (imgs, targets, anns) in enumerate(val_loader):
                num_batches += 1

                x = make_batch(imgs).to(device)
                y = make_batch(targets).to(device)

                predict = model(x)

                loss, loss_coord, loss_obj, loss_no_obj, loss_class = loss_func(predict=predict, target=y)

                val_loss += loss.detach().cpu().item()
                val_loss_coord += loss_coord.detach().cpu().item()
                val_loss_obj += loss_obj.detach().cpu().item()
                val_loss_no_obj += loss_no_obj.detach().cpu().item()
                val_loss_class += loss_class.detach().cpu().item()

                del x, y, predict, loss

            val_loss /= num_batches
            val_loss_coord /= num_batches
            val_loss_obj /= num_batches
            val_loss_no_obj /= num_batches
            val_loss_class /= num_batches

            val_loss_list.append(val_loss)
            val_loss_coord_list.append(val_loss_coord)
            val_loss_obj_list.append(val_loss_obj)
            val_loss_no_obj_list.append(val_loss_no_obj)
            val_loss_class_list.append(val_loss_class)

            print('        <val_loss> {: <10.5f}  <val_loss_coord> {: <10.5f}  <val_loss_obj> {: <10.5f}  <val_loss_class> {: <10.5f}'.format(
                val_loss_list[-1], val_loss_coord_list[-1], val_loss_obj_list[-1], val_loss_no_obj_list[-1], val_loss_class_list[-1]))

            if (e + 1) % model_save_term == 0:
                save_pth = 'saved models/{}_{}_{}epoch_{}lr_{:.5f}loss_{:.5f}losscoord_{:.5f}lossobj_{:.5f}lossnoobj_{:.5f}losscls.pth'.format(
                    model_name, dset_name, e + 1, learning_rate, val_loss_list[-1], val_loss_coord_list[-1],
                    val_loss_obj_list[-1], val_loss_no_obj_list[-1], val_loss_class_list[-1])
                torch.save(model.state_dict(), save_pth)

    x_axis = [i for i in range(len(train_loss_list))]

    plt.figure(0)
    plt.plot(x_axis, train_loss_list, 'r-', label='Train')
    plt.plot(x_axis, val_loss_list, 'b-', label='Validation')
    plt.title('Train/Validation loss')
    plt.legend()

    plt.show()
























































