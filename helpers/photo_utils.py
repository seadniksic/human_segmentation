import numpy as np
import skimage as sk
import math
import cv2
import matplotlib.pyplot as plt
import os
import pickle

def aggregate_downsample(img, target_size):

    curr_val = max(img.shape)

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

    return output_img

def inv_aggregate_downsample(img, target_size):
    pass



if __name__ == "__main__":
    # img = cv2.imread("..\\data\\AHP\\AHP\\train\\Annotations\\COCO_train2014_000000000431_sp_00.png")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # downsample_target = 256    

    with open("..\\data\\AHP\\train_annotations.pkl", "rb") as f:
        sample_names = pickle.load(f)
    # aggregate_downsample(np.array(img), downsample_target)
    img_dir = "..\\data\\AHP\\AHP\\train\\JPEGImages"
    gt_dir = "..\\data\\AHP\\AHP\\train\\Annotations"

    idx = 1

    print(type(sample_names[idx]))

    img_path = os.path.join(img_dir, sample_names[idx]) + ".jpg"
    annotation_path = os.path.join(gt_dir, sample_names[idx]) + ".png"

    print(img_path)
    
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    label = cv2.imread(annotation_path)

    image = aggregate_downsample(image, 256)
    label = aggregate_downsample(label, 256)

    fig, ax = plt.subplots(1,2)
    ax[0].imshow(image)
    ax[1].imshow(label)

    plt.show()

    print(image.shape)
    print(label.shape)

