import torch
import numpy as np
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F

from models.darknet53_ds import *

from PIL import Image

if __name__ == '__main__':
    a = torch.Tensor([-12])
    b = torch.tanh(a)
    print(a, b)