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

TRAIN_MAPPING = "data\\AHP\\train_annotations_augmented.pkl"
TEST_MAPPING = "data\\AHP\\test_annotations_augmented.pkl"
IMG_DIR = "data\\AHP\\AHP\\train\\Processed_Images"
GT_DIR = "data\\AHP\\AHP\\train\\Processed_Annotations"

start_lr = .001

train_dataset = AHP(TRAIN_MAPPING, IMG_DIR, GT_DIR)
test_dataset = AHP(TEST_MAPPING, IMG_DIR, GT_DIR)

model_top_level_name = "UNet"
model_specific_name = "batchnorm_augmented_data"
model_name = model_top_level_name + "_" + model_specific_name

log_dir = "long_trains"
experiment_name = "1"
log_experiment_name = model_name + "_" + experiment_name

custom_model = UNet()

model = sml.SupervisedMLFramework(model_name+experiment_name, custom_model, train_dataset, test_dataset, init_weights=True, 
                                  batch_size=18, log_dir=os.path.join(log_dir, log_experiment_name))

optimizer = torch.optim.AdamW(model.model.parameters(), lr=start_lr)
#scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, threshold=.0001)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=.5)
#scheduler = None

criterion = nn.CrossEntropyLoss()

train_params = {"epochs": 300, "loss_function": criterion,
         "optimizer": optimizer, "scheduler": scheduler,
         "save_dir": "checkpoints\\long_trains\\"}

model.train(**train_params)



