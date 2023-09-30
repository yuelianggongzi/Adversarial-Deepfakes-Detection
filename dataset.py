# vgg_dataset.py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "5,6"
import os
from PIL import Image
from torch import tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision import models, transforms
import cv2
from PIL import Image
from torchvision.transforms import ToTensor
import numpy as np
import dlib


class MyDatasetPhase(Dataset):
    def __init__(self, type, img_size, data_dir, transform):
        self.name2label = {"PG": 0, "CG": 1}
        self.transform = transform
        self.img_size = img_size
        self.data_dir = data_dir
        self.data_list = list()
        for file in os.listdir(self.data_dir):
            self.data_list.append(os.path.join(self.data_dir, file))
        print("Load {} Data Successfully!".format(type))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        file = self.data_list[item]
        img = Image.open(file)
        img = img.convert("RGB")
        img=tf(img)
        if ((os.path.basename(file).find('w_') == -1) and (os.path.basename(file).find('b_') == -1)):
            label = 0
        else:
            label = 1
        label = tensor(label)
        return img, label


tf=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

