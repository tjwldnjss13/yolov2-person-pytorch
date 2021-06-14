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
# from dataset.yolo_dataset import *
from dataset.yolo_person_dataset import *
from dataset.augment import *
from models.yolov2_model import YOLOV2
from loss import *
from early_stopping import EarlyStopping


def get_coco_dataset(root):
    dset_name = 'coco2017'
    transform_og = transforms.Compose([transforms.Resize((416, 416)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

    train_dset = YoloCOCODataset(root=root,
                                 image_size=(416, 416),
                                 for_train=True,
                                 transform=transform_og,
                                 augmentation=[horizontal_flip_augmentation, rotate2d_augmentation, shift_augmentation])
    val_dset = YoloCOCODataset(root=root,
                               image_size=(416, 416),
                               for_train=False,
                               transform=transform_og)

    num_classes = train_dset.num_classes
    collate_fn = custom_collate_fn

    return dset_name, train_dset, val_dset, collate_fn


def update_learning_rate(optimizer, current_epoch):
    for g in optimizer.param_groups:
        if current_epoch == 0:
            g['lr'] = .0001
        elif current_epoch == 10:
            g['lr'] = .0001
        elif current_epoch == 60:
            g['lr'] = .00001


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Define hyper parameters, parsers
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, required=False, default=16)
    parser.add_argument('--lr', type=float, required=False, default=.0001)
    parser.add_argument('--weight_decay', type=float, required=False, default=.0005)
    parser.add_argument('--momentum', type=float, required=False, default=.9)
    parser.add_argument('--num_epochs', type=int, required=False, default=100)

    args = parser.parse_args()

    batch_size = args.batch_size
    learning_rate = args.lr
    weight_decay = args.weight_decay
    momentum = args.momentum
    num_epochs = args.num_epochs

    model_save_term = 1

    # Generate COCO dataset
    root = 'D://DeepLearningData/COCO/'
    dset_name, train_dset, val_dset, collate_fn = get_coco_dataset(root)
    num_classes = 2

    # Generate VOC dataset
    # root = 'D://DeepLearningData/VOC2012'
    # dset_name, train_dset, val_dset, collate_fn = get_voc_dataset(root)
    # num_classes = 20

    # Generate data loaders
    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=custom_collate_fn)

    # Load model
    model_name = 'darknet19'
    model = YOLOV2(in_size=(416, 416), num_classes=num_classes).to(device)
    state_dict_pth = None
    # state_dict_pth = 'pretrained models/pretrain_0.0001lr_4.618loss(train)_4.734loss_0.974loss(coord)_1.298loss(obj)_1.106loss(noobj)_1.356loss(cls).pth'
    if state_dict_pth is not None:
        model.load_state_dict(torch.load(state_dict_pth), strict=False)

    # Define optimizer, loss function
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = optim.SGD(params=model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    loss_func = YoloLoss(num_classes=2)

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

        update_learning_rate(optimizer, e)
        cur_lr = optimizer.param_groups[0]['lr']

        t_train_start = time.time()
        for i, (imgs, targets) in enumerate(train_loader):
            t_batch_start = time.time()
            num_batches += 1
            num_datas += len(imgs)
            num_iter += 1
            print('[{}/{}] '.format(e + 1, num_epochs), end='')
            print(f'({num_iter}) ', end='')
            print('{}/{} '.format(num_datas, len(train_dset)), end='')
            print(f'<lr> {cur_lr}  ', end='')

            x = make_batch(imgs).to(device)
            y = make_batch(targets).to(device)

            predict = model(x)

            optimizer.zero_grad()
            loss, loss_coord, loss_obj, loss_no_obj, loss_class = loss_func(predict=predict, target=y)

            loss.backward()
            optimizer.step()

            train_loss += loss.detach().cpu().item()
            train_loss_coord += loss_coord.detach().cpu().item()
            train_loss_obj += loss_obj.detach().cpu().item()
            train_loss_no_obj += loss_no_obj.detach().cpu().item()
            train_loss_class += loss_class.detach().cpu().item()

            t_batch_end = time.time()

            H, M, S = time_calculator(t_batch_end - t_start)

            print(f'<loss> {loss.detach().cpu().item():<5.3f} ({train_loss / num_batches:<5.3f})  ', end='')
            print(f'<loss_coord> {loss_coord.detach().cpu().item():<5.3f} ({train_loss_coord / num_batches:<5.3f})  ', end='')
            print(f'<loss_obj> {loss_obj.detach().cpu().item():<5.3f} ({train_loss_obj / num_batches:<5.3f})  ', end='')
            print(f'<loss_no_obj> {loss_no_obj.detach().cpu().item():<5.3f} ({train_loss_no_obj / num_batches:<5.3f})  ', end='')
            print(f'<loss_class> {loss_class.detach().cpu().item():<5.3f} ({train_loss_class / num_batches:<5.3f})  ', end='')
            print(f'<time> {int(H):02d}:{int(M):02d}:{int(S):02d}')

            # if num_iter > 0 and num_iter % 5000 == 0:
            #     save_dir = './saved models/'
            #     if not os.path.exists(save_dir):
            #         os.mkdir(save_dir)
            #     save_pth = 'saved models/ckp_{}_{}_{}epoch_{}iter_{}lr.pth'.format(
            #         model_name, dset_name, e + 1, num_iter, learning_rate)
            #     torch.save(model.state_dict(), save_pth)

            del x, y, predict, loss, loss_coord, loss_obj, loss_no_obj, loss_class
            torch.cuda.empty_cache()

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

        print('        <train_loss> {: <.5f}  <train_loss_coord> {: <.5f}  <train_loss_obj> {: <.5f}  <train_loss_no_obj> {: 8.5f}  <train_loss_class> {: <.5f}  '.format(
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

            for i, (imgs, targets) in enumerate(val_loader):
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

                del x, y, predict, loss, loss_coord, loss_obj, loss_no_obj, loss_class
                torch.cuda.empty_cache()

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
                save_pth = 'saved models/{}_{}epoch_{}lr_{:.3f}loss(train)_{:.3f}loss_{:.3f}loss(coord)_{:.3f}loss(obj)_{:.3f}loss(noobj)_{:.3f}loss(cls).pth'.format(
                    model_name, e + 1, cur_lr, train_loss_list[-1], val_loss_list[-1], val_loss_coord_list[-1],
                    val_loss_obj_list[-1], val_loss_no_obj_list[-1], val_loss_class_list[-1])
                torch.save(model.state_dict(), save_pth)

    x_axis = [i for i in range(len(train_loss_list))]

    plt.figure(0)
    plt.plot(x_axis, train_loss_list, 'r-', label='Train')
    plt.plot(x_axis, val_loss_list, 'b-', label='Validation')
    plt.title('Train/Validation loss')
    plt.legend()

    plt.show()
























































