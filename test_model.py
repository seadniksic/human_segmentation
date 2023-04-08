import cv2
import numpy as np
import torch
from custom_models.unet_batchnorm import UNet

import torch
from AHP_Dataset import AHP_Dataset as AHP
from custom_models.SS_v1 import SS_v1
import SupervisedMLFramework as sml
import matplotlib.pyplot as plt
from custom_models.unet_batchnorm import UNet
import time
from helpers.photo_utils import aggregate_downsample, aggregate_upsample

TRAIN_MAPPING = "data\\AHP\\train_annotations.pkl"
TEST_MAPPING = "data\\AHP\\test_annotations.pkl"
IMG_DIR = "data\\AHP\\AHP\\train\\Processed_Images"
GT_DIR = "data\\AHP\\AHP\\train\\Processed_Annotations"


file_path = "C:\\Users\\sayba\\Documents\\University\\Spring_2023_T8\\1896\\human_segmentation\\checkpoints\\test1\\UNet_Standard_BatchNorm.pt"

model_save = torch.load(file_path)

model = UNet()
model.load_state_dict(model_save['model_state_dict'])

model = sml.SupervisedMLFramework("eval", model, None, None, init_weights=False, batch_size=32) 


vid = cv2.VideoCapture(0)

while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    image, data = aggregate_downsample(frame, 256)

    image = torch.as_tensor(image).permute(2,0,1) / 255

    prediction = model.predict(torch.unsqueeze(image, dim=0)).numpy()

    output = aggregate_upsample(prediction, frame.shape[0:2], 256)

    output = frame[:, :, 0] + (output * 128)  # in red channel add 128 to pixels that are human

    print(prediction.shape)
      
    # Display the resulting frame
    cv2.imshow('frame', frame)

    break
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()