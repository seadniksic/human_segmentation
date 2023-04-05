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
#from models.v1 import SGSC_PoseNet

TRAIN_MAPPING = "data\\AHP\\train_annotations.pkl"
TEST_MAPPING = "data\\AHP\\test_annotations.pkl"
IMG_DIR = "data\\AHP\\AHP\\train\\JPEGImages"
GT_DIR = "data\\AHP\\AHP\\train\\Annotations"

batch_size = 24
start_lr = .00025
end_lr = .00001

transform_list = [pu.aggregate_downsample, ToTensor]

train_dataset = AHP(TRAIN_MAPPING, IMG_DIR, GT_DIR, transform_list)
test_dataset = AHP(TEST_MAPPING, IMG_DIR, transform_list)

model_name = "initial_test"
custom_model = SGSC_PoseNet(4, 144, 16)

model = sml.SupervisedMLFramework(model_name, custom_model, train_dataset, test_dataset)

optimizer = torch.optim.Adam(model.model.parameters())
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[75, 100, 150], gamma=.00008)

criterion = model.model.calc_loss()

train_params = {"epochs": 200, "loss_function": criterion,
         "optim": optimizer, "scheduler": scheduler, "batch_size": 32,
         "save_dir": "checkpoints\\test\\"}

model.train(**train_params)














