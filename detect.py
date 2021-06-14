import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as transforms

from PIL import Image

from models.yolov2_model import YOLOV2Mobile
from utils.pytorch_util import convert_box_from_hw_to_yx, non_maximum_suppression
from utils.yolov2_tensor_generator import get_output_anchor_box_tensor

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def detect_objects(image_path, model, anchor_boxes):
    """
    :param image_path: str, path of image
    :param model: torch.nn.Module
    :param anchor_boxes: tensor, [13, 13, 4 * num bounding boxes
    :return:
    """
    img = Image.open(image_path)
    img_tensor = transforms.Compose([transforms.Resize((416, 416)), transforms.ToTensor()])(img).unsqueeze(0).to(device)
    img = np.array(img)
    H, W = img.shape[:2]

    output = model(img_tensor)
    bboxes, confidences, classes = get_object_informations(output, anchor_boxes, 5, 92)
    idx_valid = torch.where(confidences > .5)
    bboxes = bboxes[idx_valid]
    confidences = confidences[idx_valid]
    classes = classes[idx_valid]
    bboxes = convert_box_from_hw_to_yx(bboxes)
    bboxes = convert_to_original_size(bboxes, H, W)

    bbox_cat_list, conf_cat_list = categorize_informations(bboxes, confidences, classes, 92)
    bbox_list = []

    for i in range(len(bbox_cat_list)):
        if len(bbox_cat_list[i]) == 0:
            continue
        final_bboxes = non_maximum_suppression(bbox_cat_list[i], conf_cat_list[i], .5)
        bbox_list += final_bboxes

    # img = img.permute(1, 2, 0)
    for i, bbox in enumerate(bbox_list):
        y1, x1, y2, x2 = int(bbox[0].item()), int(bbox[1].item()), int(bbox[2].item()), int(bbox[3].item())
        cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

    cv.imshow('detection', img)
    if cv.waitKey(0) == ord('q'):
        cv.destroyAllWindows()



def get_object_informations(tensor, anchor_boxes, num_predict_bounding_boxes, num_classes):
    """
    :param tensor: tensor, [1, 13, 13, (5 + num classes) * num bounding boxes]
    :param anchor_boxes: tensor, [13, 13, 4 * num bounding boxes]
    :param num_predict_bounding_boxes: int
    :param num_classes: int
    :return:
    """
    out = tensor
    num_bbox = num_predict_bounding_boxes

    out = out.squeeze(0).reshape(-1, 5 + num_classes)
    bboxes = out[:, :4]
    confidences = torch.nn.Sigmoid()(out[:, 4])
    classes = torch.nn.Softmax(dim=1)(out[:, 5:]).argmax(dim=1)

    anchor_boxes = anchor_boxes.reshape(-1, 4)

    bboxes[:, :2] = torch.nn.Sigmoid()(bboxes[:, :2]) + anchor_boxes[:, :2]
    # for i in range(bboxes.shape[0]):
    #     print(torch.exp(bboxes[i, 2:4]), anchor_boxes[i, 2:4])
    bboxes[:, 2:4] = torch.exp(bboxes[:, 2:4]) * anchor_boxes[:, 2:4]

    return bboxes, confidences, classes


def convert_to_original_size(bounding_boxes, height, width):
    """
    :param bounding_boxes: tensor, [13^2, 4]
    :param height: int, height of original image
    :param width: int, width of original image
    :return:
    """
    bboxes = bounding_boxes

    bboxes[:, 0] = bboxes[:, 0] * (height / 13)
    bboxes[:, 1] = bboxes[:, 1] * (width / 13)
    bboxes[:, 2] = bboxes[:, 2] * (height / 13)
    bboxes[:, 3] = bboxes[:, 3] * (width / 13)

    return bboxes


def categorize_informations(bounding_boxes, confidences, classes, num_classes):
    """
    :param bounding_boxes: tensor, [13 ^ 2, 4]
    :param confidences: tensor, [13 ^ 2]
    :param classes: tensor, [13 ^ 2]
    :param num_classes: int
    :return:
    """
    bboxes = bounding_boxes

    bbox_category_list = [None for _ in range(num_classes)]
    confidence_category_list = [None for _ in range(num_classes)]

    for i in range(num_classes):
        idx = torch.where(classes == i)[0]
        bbox_category_list[i] = bboxes[idx]
        confidence_category_list[i] = confidences[idx]

    return bbox_category_list, confidence_category_list


if __name__ == '__main__':
    anchor_box_samples = torch.Tensor([[1.73145, 1.3221],
                                       [4.00944, 3.19275],
                                       [8.09892, 5.05587],
                                       [4.84053, 9.47112],
                                       [10.0071, 11.2364]])
    anchor_box_base = get_output_anchor_box_tensor(anchor_box_sizes=anchor_box_samples,
                                                   out_size=(13, 13)).to(device)

    state_dict_pth = 'pretrained models/yolov2mobile_coco2017_1epoch_2fold_0.0001lr_12.81251loss_3.76018losscoord_6.30102lossconf_2.75131losscls.pth'
    model = YOLOV2Mobile((416, 416), 92, anchor_box_samples).to(device)
    model.load_state_dict(torch.load(state_dict_pth))
    model.eval()

    img_pth = 'sample/boat.jpg'

    with torch.no_grad():
        detect_objects(img_pth, model, anchor_box_base)



























