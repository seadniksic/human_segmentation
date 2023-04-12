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
from helpers.photo_utils import aggregate_downsample, aggregate_upsample, interpolate_downsample
import torch_tensorrt

TRAIN_MAPPING = "data\\AHP\\train_annotations.pkl"
TEST_MAPPING = "data\\AHP\\test_annotations.pkl"
IMG_DIR = "data\\AHP\\AHP\\train\\Processed_Images"
GT_DIR = "data\\AHP\\AHP\\train\\Processed_Annotations"


file_path = "compiled_tensorrt\\UNet.ts"

trt_ts_module = torch.jit.load(file_path)

vid = cv2.VideoCapture(0)
vid.set(3, 1920)
vid.set(4, 1080)
base_now=0

total_time = 0
downsample_time = 0
upsample_time = 0
inference_time = 0
frames = 0

while(True):

    ret, frame = vid.read()

    image, data = interpolate_downsample(frame, 256)

    image = torch.as_tensor(image).permute(2,0,1) / 255

    prediction = trt_ts_module(torch.unsqueeze(image, dim=0))
    
    prediction =  torch.argmax(prediction, dim=1).squeeze().cpu().numpy()

    output = aggregate_upsample(prediction, frame.shape[0:2], 256)

    frame[:,:,2][output > 0] = 255  # in red channel add 128 to pixels that are human

    # Display the resulting frame
    cv2.imshow('frame', output.astype(np.uint8))
    
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

