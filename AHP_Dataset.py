import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import cv2
import helpers.photo_utils as pu
import torch
import matplotlib.pyplot as plt


class AHP_Dataset(Dataset):
    def __init__(self, name_mapping, img_dir, gt_dir):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        with open(name_mapping, "rb") as f:
            self.sample_names = pickle.load(f)

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.sample_names[idx]) + ".jpg"
        annotation_path = os.path.join(self.gt_dir, self.sample_names[idx]) + ".png"
        
        image = cv2.imread(img_path)
        label = cv2.imread(annotation_path)

        image = torch.as_tensor(image).permute(2,0,1).float()
        label = torch.as_tensor(label[:,:,0]).long()

        image = image / 255 #normalize image pixel values   
        
        return image, label
