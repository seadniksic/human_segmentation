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
from custom_models.unet_better import UNet
from custom_models.SS_v1 import SS_v1

TRAIN_MAPPING = "data\\AHP\\train_annotations.pkl"
TEST_MAPPING = "data\\AHP\\test_annotations.pkl"
IMG_DIR = "data\\AHP\\AHP\\train\\Processed_Images"
GT_DIR = "data\\AHP\\AHP\\train\\Processed_Annotations"

start_lr = .01

train_dataset = AHP(TRAIN_MAPPING, IMG_DIR, GT_DIR)
test_dataset = AHP(TEST_MAPPING, IMG_DIR, GT_DIR)

model_name = "initial_test_Unet"
custom_model = UNet()

model = sml.SupervisedMLFramework(model_name, custom_model, train_dataset, test_dataset, batch_size=64)

optimizer = torch.optim.AdamW(model.model.parameters(), lr=start_lr)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

criterion = nn.CrossEntropyLoss()

train_params = {"epochs": 100, "loss_function": criterion,
         "optimizer": optimizer, "scheduler": scheduler,
         "save_dir": "checkpoints\\test1\\"}

model.train(**train_params)
















