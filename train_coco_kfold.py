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

from torch.utils.data import DataLoader, ConcatDataset, Subset, SubsetRandomSampler
from PIL import Image
from sklearn.model_selection import KFold

from utils.util import time_calculator
from utils.pytorch_util import make_batch
from utils.yolov2_tensor_generator import get_output_anchor_box_tensor, get_yolo_v2_output_tensor, get_yolo_v2_target_tensor
from dataset.coco_dataset import COCODataset, custom_collate_fn
from dataset.augment import GaussianNoise
from models.yolov2_model import YOLOV2Mobile
from loss import yolov2_custom_loss_1 as loss_func
from early_stopping import EarlyStopping


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
    dset_name = 'coco2017'
    root = 'D://DeepLearningData/COCOdataset2017/'
    train_img_dir = os.path.join(root, 'images', 'train')
    val_img_dir = os.path.join(root, 'images', 'val')
    train_ann_pth = os.path.join(root, 'annotations', 'instances_train2017.json')
    val_ann_pth = os.path.join(root, 'annotations', 'instances_val2017.json')

    transform_og = transforms.Compose([transforms.Resize((416, 416)),
                                       transforms.ToTensor()])
    transform_norm = transforms.Compose([transforms.Resize((416, 416)),
                                         transforms.ToTensor(),
                                         transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    transform_noise = transforms.Compose([transforms.Resize((416, 416)),
                                          transforms.ToTensor(),
                                          GaussianNoise(mean=0, std=.2)])
    transform_norm_noise = transforms.Compose([transforms.Resize((416, 416)),
                                               transforms.ToTensor(),
                                               transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
                                               GaussianNoise(mean=0, std=.2)])
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

    dset = COCODataset(root=root, images_dir=train_img_dir, annotation_path=train_ann_pth, is_categorical=True,
                             transforms=transform_og, image_size=(416, 416))
    # val_dset = COCODataset(root=root, images_dir=val_img_dir, annotation_path=val_ann_pth, is_categorical=True,
    #                          transforms=transforms)
    # dset = ConcatDataset([train_dset, val_dset])

    num_classes = dset.num_classes
    n_data = len(dset)

    # Define K-fold validator
    n_split = 5
    kfold = KFold(n_splits=n_split, shuffle=True)

    # Load model
    model_name = 'yolov2mobile'
    anchor_box_samples = torch.Tensor([[1.73145, 1.3221],
                                       [4.00944, 3.19275],
                                       [8.09892, 5.05587],
                                       [4.84053, 9.47112],
                                       [10.0071, 11.2364]])
    model = YOLOV2Mobile(in_size=(416, 416), num_classes=num_classes, anchor_box_samples=anchor_box_samples).to(device)
    state_dict_pth = None
    state_dict_pth = 'pretrained models/yolov2mobile_coco2017_1epoch_2fold_0.0001lr_12.81251loss_3.76018losscoord_6.30102lossconf_2.75131losscls.pth'
    if state_dict_pth is not None:
        model.load_state_dict(torch.load(state_dict_pth), strict=False)

    # Define optimizer, loss function
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=.9)

    # Define anchor box configuration
    dummy = torch.zeros(1, 3, 416, 416).to(device)
    pred_dummy = model(dummy)

    anchor_box_base = get_output_anchor_box_tensor(anchor_box_sizes=anchor_box_samples, out_size=pred_dummy.shape[1:3]).to(device)

    train_loss_list = []
    train_loss_coord_list = []
    train_loss_confidence_list = []
    train_loss_class_list = []
    val_loss_list = []
    val_loss_coord_list = []
    val_loss_confidence_list = []
    val_loss_class_list = []
    model.train()

    t_start = time.time()
    for e in range(num_epochs):
        num_datas = 0
        train_loss = 0
        train_loss_coord = 0
        train_loss_confidence = 0
        train_loss_class = 0
        num_train_batches = 0
        num_val_batches = 0

        for fold, (train_ids, val_ids) in enumerate(kfold.split(dset)):
            train_subsampler = SubsetRandomSampler(train_ids)
            val_subsampler = SubsetRandomSampler(val_ids)

            train_loader = DataLoader(dset, batch_size=batch_size, sampler=train_subsampler, collate_fn=custom_collate_fn)
            val_loader = DataLoader(dset, batch_size=batch_size, sampler=val_subsampler, collate_fn=custom_collate_fn)

            num_datas = 0
            t_train_start = time.time()
            for i, (imgs, anns) in enumerate(train_loader):
                t_batch_start = time.time()
                num_train_batches += 1
                num_datas += len(imgs)
                print(f'[{e + 1}/{num_epochs}] ', end='')
                print(f'({fold + 1}/{n_split} FOLD) ', end='')
                print(f'{num_datas}/{int(len(dset) * (1 - 1 / n_split))}  ', end='')

                x = make_batch(imgs).to(device)

                predict_temp = model(x)
                predict_list = []
                y_list = []

                for b in range(len(anns)):
                    ground_truth_box = anns[b]['bbox']
                    label_categorical = anns[b]['label_categorical']

                    # print(f'{label_int} {label_categorical} {name} {fn} {ground_truth_box / 32}')

                    h_img, w_img = 416, 416
                    ratio_y, ratio_x = 1 / 32, 1 / 32
                    ground_truth_box = torch.as_tensor(ground_truth_box)
                    if len(ground_truth_box.shape) < 2:
                        ground_truth_box = ground_truth_box.unsqueeze(0)

                    predict_list.append(get_yolo_v2_output_tensor(predict_temp[b], anchor_box_base))
                    y_temp = get_yolo_v2_target_tensor(ground_truth_boxes=ground_truth_box,
                                                        anchor_boxes=anchor_box_base,
                                                        labels=label_categorical,
                                                        n_bbox_predict=5,
                                                        n_class=num_classes,
                                                        in_size=(h_img, w_img),
                                                        out_size=(13, 13))
                    y_list.append(y_temp)

                y = make_batch(y_list).to(device)
                predict = make_batch(predict_list).to(device)

                del x, predict_temp, predict_list, y_list, y_temp

                optimizer.zero_grad()
                loss, loss_coord, loss_confidence, loss_class = loss_func(predict=predict, target=y, anchor_boxes=anchor_box_base, num_bbox_predict=5, num_classes=num_classes)
                loss.backward()
                optimizer.step()

                train_loss += loss.detach().cpu().item()
                train_loss_coord += loss_coord.detach().cpu().item()
                train_loss_confidence += loss_confidence.detach().cpu().item()
                train_loss_class += loss_class.detach().cpu().item()

                t_batch_end = time.time()

                H, M, S = time_calculator(t_batch_end - t_start)

                print('<loss> {: <10.5f}  <loss_coord> {: <10.5f}  <loss_confidence> {: <10.5f}  <loss_class> {: <10.5f}  '.format(
                    loss.detach().cpu().item(), loss_coord.detach().cpu().item(),
                    loss_confidence.detach().cpu().item(), loss_class.detach().cpu().item()), end='')
                print('<loss_avg> {: <10.5f}  <loss_coord_avg> {: <10.5f}  <loss_confidence_avg> {: <10.5f}  <loss_class_avg> {: <10.5f}  '.format(
                    train_loss / num_train_batches, train_loss_coord / num_train_batches, train_loss_confidence / num_train_batches,
                    train_loss_class / num_train_batches
                ), end='')
                print('<time> {:02d}:{:02d}:{:02d}'.format(int(H), int(M), int(S)))

                del y, predict, loss

            with torch.no_grad():
                model.eval()
                val_loss = 0
                val_loss_coord = 0
                val_loss_confidence = 0
                val_loss_class = 0

                for i, (imgs, anns) in enumerate(val_loader):
                    num_val_batches += 1

                    x = make_batch(imgs).to(device)

                    predict_temp = model(x)
                    predict_list = []
                    y_list = []

                    for b in range(len(anns)):
                        h_img, w_img = 416, 416
                        ground_truth_box = anns[b]['bbox']
                        label = anns[b]['label_categorical']

                        ratio_h, ratio_w = 1 / 32, 1 / 32
                        ground_truth_box = torch.as_tensor(ground_truth_box)
                        if len(ground_truth_box.shape) < 2:
                            ground_truth_box = ground_truth_box.unsqueeze(0)

                        predict_list.append(get_yolo_v2_output_tensor(predict_temp[b], anchor_box_base))
                        y_list.append(get_yolo_v2_target_tensor(ground_truth_boxes=ground_truth_box,
                                                                labels=label,
                                                                anchor_boxes=anchor_box_base,
                                                                n_bbox_predict=5,
                                                                n_class=num_classes,
                                                                in_size=(h_img, w_img),
                                                                out_size=(13, 13)))

                    y = make_batch(y_list).to(device)
                    predict = make_batch(predict_list).to(device)

                    del predict_temp, predict_list, y_list

                    loss, loss_coord, loss_confidence, loss_class = loss_func(predict=predict, target=y, anchor_boxes=anchor_box_base, num_bbox_predict=5, num_classes=num_classes)

                    val_loss += loss.detach().cpu().item()
                    val_loss_coord += loss_coord.detach().cpu().item()
                    val_loss_confidence += loss_confidence.detach().cpu().item()
                    val_loss_class += loss_class.detach().cpu().item()

                    del x, y, predict, loss

                print(f'\t\t<val_loss> {val_loss / num_val_batches: <10.5f}  <val_loss_coord> {val_loss_coord /num_val_batches: <10.5f}  '
                      f'<val_loss_confidence> {val_loss_confidence / num_val_batches: <10.5f}  <val_loss_class> {val_loss_class /num_val_batches: <10.5f}')

                if (e + 1) % model_save_term == 0:
                    save_pth = f'saved models/{model_name}_{dset_name}_{e + 1}epoch_{fold + 1}fold_{learning_rate}lr_{val_loss / num_val_batches:.5f}loss_' \
                               f'{val_loss_coord / num_val_batches:.5f}losscoord_{val_loss_confidence / num_val_batches:.5f}lossconf_{val_loss_class / num_val_batches:.5f}losscls.pth'
                    torch.save(model.state_dict(), save_pth)

                train_loss_list.append(train_loss / num_train_batches)
                train_loss_coord_list.append(train_loss_coord / num_train_batches)
                train_loss_confidence_list.append(train_loss_confidence / num_train_batches)
                train_loss_class_list.append(train_loss_class / num_train_batches)

                val_loss_list.append(val_loss / num_val_batches)
                val_loss_coord_list.append(val_loss_coord / num_val_batches)
                val_loss_confidence_list.append(val_loss_confidence / num_val_batches)
                val_loss_class_list.append(val_loss_class / num_val_batches)

    x_axis = [i for i in range(len(train_loss_list))]

    plt.figure(0)
    plt.plot(x_axis, train_loss_list, 'r-', label='Train')
    plt.plot(x_axis, val_loss_list, 'b-', label='Validation')
    plt.title('Train/Validation loss')
    plt.legend()

    plt.show()
























































