import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import SupervisedMLFramework as sml
import helpers.photo_utils as pu
from AHP_Dataset import AHP_Dataset as AHP
from custom_models.unet_batchnorm import UNet
from custom_models.SS_v1 import SS_v1
import os


TRAIN_MAPPING = "data\\AHP\\train_annotations.pkl"
TEST_MAPPING = "data\\AHP\\test_annotations.pkl"
IMG_DIR = "data\\AHP\\AHP\\train\\Raw_Downsample_Images"
GT_DIR = "data\\AHP\\AHP\\train\\Processed_Annotations"

train_dataset = AHP(TRAIN_MAPPING, IMG_DIR, GT_DIR)
test_dataset = AHP(TEST_MAPPING, IMG_DIR, GT_DIR)

custom_model = UNet()

criterion = nn.CrossEntropyLoss()

model = sml.SupervisedMLFramework(model_name="", model=custom_model, train_dataset=train_dataset, test_dataset=test_dataset,
                                  init_weights=True, batch_size=32, log_dir="")

file_path = "C:\\Users\\sayba\\Documents\\University\\Spring_2023_T8\\1896\\human_segmentation\\checkpoints\\long_trains\\UNet_Standard_BatchNorm3.pt"

model_save = torch.load(file_path)

model.model.load_state_dict(model_save['model_state_dict'])

testing_loss = model.test(criterion)
