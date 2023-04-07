import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as f


'''
        Type                       Shape
Input:  image patch                3x572x572
Output: pixel level segmentation   1x572x572


'''
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.down_conv1 = double_conv(3, 64)
        self.down_conv2 = double_conv(64, 128)
        self.down_conv3 = double_conv(128, 256)
        self.down_conv4 = double_conv(256, 512)
        self.down_conv5 = double_conv(512, 1024)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up_conv1 = double_conv(1024, 512)
        self.up_conv2 = double_conv(512, 256)
        self.up_conv3 = double_conv(256, 128)
        self.up_conv4 = double_conv(128, 64)
        self.up_conv5 = double_conv(64, 1)

        self.transpose_conv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0)
        self.transpose_conv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0)
        self.transpose_conv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)
        self.transpose_conv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)

        self.output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=(1,1), padding_mode='zeros')

    def forward(self, image):
        # print('image size: {}'.format(image.shape))
        down1 = self.down_conv1(image)
        down1_pool = self.max_pool(down1)
        down2 = self.down_conv2(down1_pool)
        down2_pool = self.max_pool(down2)
        down3 = self.down_conv3(down2_pool)
        down3_pool = self.max_pool(down3)
        down4 = self.down_conv4(down3_pool)
        down4_pool = self.max_pool(down4)
        down5 = self.down_conv5(down4_pool)
        
        up1 = self.transpose_conv1(down5)
        up1 = torch.cat((down4, up1), dim=1)
        up1 = self.up_conv1(up1)
 
        up2 = self.transpose_conv2(up1)
        up2 = torch.cat((down3, up2), dim=1)
        up2 = self.up_conv2(up2)

        up3 = self.transpose_conv3(up2)
        up3 = torch.cat((down2, up3), dim=1)
        up3 = self.up_conv3(up3)

        up4 = self.transpose_conv4(up3)
        up4 = torch.cat((down1, up4), dim=1)
        up4 = self.up_conv4(up4)

        output = self.output(up4)
        return output
        

   
def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, padding=(1,1), padding_mode='zeros'),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, padding=(1,1), padding_mode='zeros'),
        nn.ReLU(inplace=True)
    )
    return conv

if __name__ == '__main__':
    model = UNet()
    image = torch.rand((1,3,512,512))
    predict = model(image)
    print(predict)
    print(predict.shape)