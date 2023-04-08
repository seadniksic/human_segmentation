import numpy as np
import skimage as sk
import math
import cv2
import matplotlib.pyplot as plt
import os
import pickle
import torch.nn as nn
import torch
import scipy.ndimage

def aggregate_downsample(img, target_size):

    curr_val = max(img.shape)

    print(img.shape)

    downsample_factor = math.ceil(curr_val / target_size)
    #convert image largest_dim to leading_dimension size
    downsampled_image = sk.measure.block_reduce(img, (downsample_factor, downsample_factor, 1), np.mean).astype(int)

    height_downsampled, width_downsampled  = downsampled_image.shape[:-1]

    #new target image to fill in
    output_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)

    #pad on both dimensions to ensure that it's 256x256
    total_padding_rows = target_size - downsampled_image.shape[0]
    total_padding_cols = target_size - downsampled_image.shape[1]

    top_padding = int(total_padding_rows / 2)
    left_padding = int(total_padding_cols / 2)

    output_img[top_padding:top_padding+height_downsampled, left_padding:left_padding+width_downsampled, :] = downsampled_image

    # plt.imshow(output_img)
    # plt.show()

    return output_img, (height_downsampled, width_downsampled)



def aggregate_upsample(cropped_img, orig_size, orig_target_size):

    curr_val = max(orig_size)

    upsample_factor = math.ceil(curr_val / orig_target_size)

    height_downsampled, width_downsampled = int(orig_size[0] / upsample_factor), int(orig_size[1] / upsample_factor)

    total_padding_rows = orig_target_size - height_downsampled
    total_padding_cols = orig_target_size - width_downsampled

    top_padding = int(total_padding_rows / 2)
    left_padding = int(total_padding_cols / 2)

    temp_array = cropped_img[top_padding:top_padding+height_downsampled, left_padding:left_padding+width_downsampled]

    temp_array = scipy.ndimage.zoom(temp_array, upsample_factor, order=1)

    height_diff = orig_size[0] - temp_array.shape[0]
    width_diff = orig_size[1] - temp_array.shape[1]

    temp_array = np.pad(temp_array, ((0, height_diff), (0, width_diff)), 'constant', constant_values=0)

    return temp_array









if __name__ == "__main__":
    # img = cv2.imread("..\\data\\AHP\\AHP\\train\\Annotations\\COCO_train2014_000000000431_sp_00.png")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # downsample_target = 256    

    with open("..\\data\\AHP\\train_annotations.pkl", "rb") as f:
        sample_names = pickle.load(f)
    # aggregate_downsample(np.array(img), downsample_target)
    img_dir = "..\\data\\AHP\\AHP\\train\\JPEGImages"
    gt_dir = "..\\data\\AHP\\AHP\\train\\Annotations"\
    
    print(sample_names)

    idx = 1

    print(sample_names[idx])

    img_path = os.path.join(img_dir, sample_names[idx]) + ".jpg"
    annotation_path = os.path.join(gt_dir, sample_names[idx]) + ".png"

    print(img_path)
    
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    label = cv2.imread(annotation_path)

    print(image.shape)
    print(label.shape)

    image = aggregate_downsample(image, 256)
    label = aggregate_downsample(label, 256)

    #Convert to ground truth mask
    #label[label > 0] = 1

    loss = nn.CrossEntropyLoss()


    #print(label[147,136])

    fig, ax = plt.subplots(1,2)
    ax[0].imshow(image)
    ax[1].imshow(label)

    # prediction = image[:,:, :2]
    # target = label[:, :, -1]
    # print(prediction.shape)

    # prediction = torch.tensor(prediction).float()
    # prediction = prediction.permute(2,0,1).unsqueeze(dim=0)
    # target = torch.tensor(target).long().unsqueeze(dim=0)
    # print(prediction.shape)
    # print(target.shape)

    # print(loss(prediction, target))
    # # target = torch.unsqueeze(target, dim=0).shape
    # print(target.dtype)

    #print(nn.functional.one_hot(target, num_classes=2).shape)

    



    plt.show()

    print(image.shape)
    print(label.shape)

