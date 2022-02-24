import torch
import torch.nn as nn

from unet_modules import DoubleConv, DownSample, UpSample


class Unet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Unet, self).__init__()
        self.down_conv_1 = DownSample(in_channels, 64)
        self.down_conv_2 = DownSample(64, 128)
        self.down_conv_3 = DownSample(128, 256)
        self.down_conv_4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(512, 1024)

        self.up_conv_1 = UpSample(1024, 512)
        self.up_conv_2 = UpSample(512, 256)
        self.up_conv_3 = UpSample(256, 128)
        self.up_conv_4 = UpSample(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)
    
    def forward(self, x):
        d1, p1 =self.down_conv_1(x)
        d2, p2 =self.down_conv_2(p1)
        d3, p3 =self.down_conv_3(p2)
        d4, p4 =self.down_conv_4(p3)

        b = self.bottle_neck(p4)

        u1 = self.up_conv_1(b, d4)
        u2 = self.up_conv_2(u1, d3)
        u3 = self.up_conv_3(u2, d2)
        u4 = self.up_conv_4(u3, d1)

        out = self.out(u4)

        return out
    
if __name__ == "__main__":
    double_conv = DoubleConv(256,256)
    print(double_conv)

    input_image = torch.rand((1,3,512,512))
    model = Unet(3,10)
    output = model(input_image)
    print(output.size()) #([1,10,512,512])
