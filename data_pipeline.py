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
IMG_DIR = "data\\AHP\\AHP\\train\\Processed_Images"
GT_DIR = "data\\AHP\\AHP\\train\\Processed_Annotations"

start_lr = .01

train_dataset = AHP(TRAIN_MAPPING, IMG_DIR, GT_DIR)
test_dataset = AHP(TEST_MAPPING, IMG_DIR, GT_DIR)

model_top_level_name = "SS_v1"
model_specific_name = "Standard_single"
model_name = model_top_level_name + "_" + model_specific_name

log_dir = "runs_single_batch"
experiment_name = "3"
log_experiment_name = model_name + "_" + experiment_name

custom_model = SS_v1()

model = sml.SupervisedMLFramework(model_name+experiment_name, custom_model, train_dataset, test_dataset, init_weights=True, 
                                  batch_size=32, log_dir=os.path.join(log_dir, log_experiment_name))

optimizer = torch.optim.AdamW(model.model.parameters(), lr=start_lr)
#scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, threshold=.0001)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=.5)
scheduler = None

criterion = nn.CrossEntropyLoss()

train_params = {"epochs": 300, "loss_function": criterion,
         "optimizer": optimizer, "scheduler": scheduler,
         "save_dir": "checkpoints\\long_trains\\"}

model.train_single_batch(**train_params)

