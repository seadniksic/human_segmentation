import cv2
import numpy as np
import torch
from custom_models.unet_batchnorm import UNet

import torch
from AHP_Dataset import AHP_Dataset as AHP
#from custom_models.SS_v1 import SS_v1
#import SupervisedMLFramework as sml
import matplotlib.pyplot as plt
from custom_models.unet_batchnorm import UNet
import time
from helpers.photo_utils import aggregate_downsample, aggregate_upsample, interpolate_downsample
from std_msgs.msg import UInt8MultiArray
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

bridge = CvBridge()

print(torch.cuda.is_available())
file_path = "models/UNet_Standard_BatchNorm3.pt"

model_save = torch.load(file_path)

model = UNet()
model.load_state_dict(model_save['model_state_dict'])

model.to('cuda')
print(f"model: {next(model.parameters()).is_cuda}")

model.eval()

#model = sml.SupervisedMLFramework("eval", model, None, None, init_weights=False, batch_size=32) 


# vid = cv2.VideoCapture(0)
# vid.set(3, 1920)
# vid.set(4, 1080)
#base_now=0

total_time = 0
downsample_time = 0
upsample_time = 0
inference_time = 0
frames = 0

newPub = rospy.Publisher("cameraFeedNetworkNodeSubscriber", UInt8MultiArray, queue_size=5)

#while(True):

# def detect_pose(frame):

#     #ret, frame = vid.read()

#     f = frame.copy()

#     image, data = interpolate_downsample(frame, 256)

#     image = torch.as_tensor(image).permute(2,0,1) / 255
#     #image.to("cuda")
#     image.cuda()
#     print(f"Image cuda: {image.is_cuda}")

#     #prediction = model.predict(torch.unsqueeze(image, dim=0), threshold=None).numpy()

    
#     #print(torch.unsqueeze(image).shape)

#     #sample = sample.to(self.device)
#     with torch.no_grad():
#         pred = model(torch.unsqueeze(image, dim=0))
#         # if softmax == True:
#         #     sm = nn.Softmax(dim=1)
#         #     pred = sm(pred)
#         # if threshold != None:
#         #         pred[pred < threshold] = 0
        
#         pred = torch.argmax(pred, dim=1).squeeze()
#     #pred.to('cpu')

#     output = aggregate_upsample(pred, frame.shape[0:2], 256)

#     f[:,:,2][output > 0] = 255  # in red channel add 128 to pixels that are human

#     return f

    #output *= 255

    # Display the resulting frame
    # cv2.imshow('frame', output.astype(np.uint8))
    
    # # the 'q' button is set as the
    # # quitting button you may use any
    # # desired button of your choice
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# frame = None
# newFrame = 0

# def image_callback(msg):
#     global frame
#     frame = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
#     global newFrame
#     newFrame = 1
#     # pose = detect_pose(frame)

    # print(pose.shape)



    #  #tart =  time.time()
    # __, sendData = cv2.imencode('.jpg', pose, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    # messagePayload = UInt8MultiArray()
    # messagePayload.data = sendData.tobytes()
    # #messagePayload = bridge.cv2_to_imgmsg(pose)
    # newPub.publish(messagePayload)
# # After the loop release the cap object
# vid.release()
# # Destroy all the windows
# cv2.destroyAllWindows()


rospy.init_node("image_subscriber")
#rospy.Subscriber("/camera/color/image_raw", Image, image_callback)

while not rospy.is_shutdown():
    r = rospy.wait_for_message("/camera/color/image_raw", Image, timeout=None)

    print("hello")
    frame = bridge.imgmsg_to_cv2(r, desired_encoding='passthrough')
    #pose = detect_pose(frame)

    f = frame.copy()

    image, data = interpolate_downsample(frame, 256)

    image = torch.as_tensor(image).permute(2,0,1) / 255
    print(type(image))
    image = image.to(device="cuda")
    #image.cuda()
    print(f"Image cuda: {image.is_cuda}")

    #prediction = model.predict(torch.unsqueeze(image, dim=0), threshold=None).numpy()

    
    #print(torch.unsqueeze(image).shape)

    #sample = sample.to(self.device)
    with torch.no_grad():
        pred = model(torch.unsqueeze(image, dim=0))
        # if softmax == True:
        #     sm = nn.Softmax(dim=1)
        #     pred = sm(pred)
        # if threshold != None:
        #         pred[pred < threshold] = 0
        
        pred = torch.argmax(pred, dim=1).squeeze()
    pred = pred.to('cpu')

    output = aggregate_upsample(pred, frame.shape[0:2], 256)

    f[:,:,2][output > 0] = 255  # in red channel add 128 to pixels that are human



    __, sendData = cv2.imencode('.jpg', f, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    messagePayload = UInt8MultiArray()
    messagePayload.data = sendData.tobytes()
    newPub.publish(messagePayload)
    newFrame = 0

#sock.close()

