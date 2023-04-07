import torch
import torch.nn as nn

def down_conv_block(in_channels, out_channels):
    block = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2)
    )

    return block

def up_conv_block(in_channels, out_channels, output_padding):
    block = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=2, dilation=2, output_padding=output_padding)
    )
    return block


## Real time Semantic Segmentation
class SS_v1(torch.nn.Module):

    def __init__(self):
        super(SS_v1, self).__init__()

        self.down1 = down_conv_block(3, 32)
        self.down2 = down_conv_block(32, 64)
        self.down3 = down_conv_block(64, 128)

        #Upsample
        self.up1 = up_conv_block(128, 64, 0)
        self.up2 = up_conv_block(64, 32, 1)
        self.up3 = up_conv_block(32, 2, 1)

        self.test1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1)
        self.test2 = nn.BatchNorm2d(32)
        self.test3 = nn.ReLU()
        self.test4 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.output = nn.Softmax(dim=1)

    def forward(self, img):
        
        x = self.down1(img)
        x = self.down2(x)
        x = self.down3(x)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)

        x = self.output(x)

        return x



if __name__ == "__main__":
    image = torch.rand((1,3,256,256))
    target = torch.rand((1,256,256)).long()
    model = SS_v1()

    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with torch.no_grad():
        y_pred = model(image)

        print(y_pred.dtype)
        print(target.dtype)
        loss = criterion(y_pred, target)
        print(loss.item())




