import torch
import torch.nn as nn

def down_conv_block(in_channels, out_channels):
    block = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1),
        nn.BatchNorm2d(),
        nn.ReLU(),
        nn.MaxPool2d()
    )

def up_conv_block(in_channels, out_channels):
     block = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1)
    )


## Real time Semantic Segmentation
class SS_v1(torch.nn.Module):

    def __init__(self):
        super(SS_v1, self).__init__()

        self.down1 = down_conv_block(3, 32)
        self.down2 = down_conv_block(32, 64)
        self.down3 = down_conv_block(64, 128)

        #Upsample
        self.up1 = up_conv_block(128, 64)
        self.up2 = up_conv_block(64, 32)
        self.up3 = up_conv_block(32, 2)

    def forward(self, img):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)

        x = nn.Softmax(dim=1)

        return x






