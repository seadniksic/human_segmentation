import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import cv2


class AHP_Dataset(Dataset):
    def __init__(self, name_mapping, img_dir, gt_dir, transform_list):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.transform_list = transform_list
        with open(name_mapping, "rb") as f:
            self.sample_names = pickle.load(f)

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.sample_names[idx]) + ".jpg"
        annotation_path = os.path.join(self.gt_dir, self.sample_names[idx]) + ".png"
        
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        label = cv2.imread(annotation_path)

        for transform in self.transform_list:
            image = transform(image)
            label = transform(label)
        
        return image, label
