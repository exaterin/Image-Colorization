import torch
import torch.nn as nn
import torch.nn.functional as F

class RefcCNNModel(nn.Module):
    def __init__(self):
        super(RefcCNNModel, self).__init__()

        def conv_block(in_channels, out_channels, kernel_size=3, stride=2, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2)
            )

        def deconv_block(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

        # Encoding layers
        self.enc1 = conv_block(4, 64)    # 256x256x4 -> 128x128x64 (4 channels: 1 grayscale + 3 RGB reference)
        self.enc2 = conv_block(64, 128)  # 128x128x64 -> 64x64x128
        self.enc3 = conv_block(128, 256) # 64x64x128 -> 32x32x256
        self.enc4 = conv_block(256, 512) # 32x32x256 -> 16x16x512
        self.enc5 = conv_block(512, 1024) # 16x16x512 -> 8x8x1024
        self.enc6 = conv_block(1024, 1024) # 8x8x1024 -> 4x4x1024

        # Decoding layers
        self.dec1 = deconv_block(1024, 1024) # 4x4x1024 -> 8x8x1024
        self.dec2 = deconv_block(2048, 512) # 8x8x2048 -> 16x16x512
        self.dec3 = deconv_block(1024, 256) # 16x16x1024 -> 32x32x256
        self.dec4 = deconv_block(512, 128) # 32x32x512 -> 64x64x128
        self.dec5 = deconv_block(256, 64)  # 64x64x256 -> 128x128x64
        self.dec6 = deconv_block(128, 3)  # 128x128x128 -> 256x256x3

    def forward(self, x, ref):
        # Concatenate the input image and the reference image along the channel dimension
        x = torch.cat((x, ref), dim=1)
        
        # Encoding
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)

        # Decoding + Concatenating with encoding layers
        d1 = self.dec1(e6)
        d1 = torch.cat((d1, e5), dim=1)
        d2 = self.dec2(d1)
        d2 = torch.cat((d2, e4), dim=1)
        d3 = self.dec3(d2)
        d3 = torch.cat((d3, e3), dim=1)
        d4 = self.dec4(d3)
        d4 = torch.cat((d4, e2), dim=1)
        d5 = self.dec5(d4)
        d5 = torch.cat((d5, e1), dim=1)
        d6 = self.dec6(d5)

        return torch.sigmoid(d6)


if __name__ == '__main__':
    model = RefcCNNModel()
    print(model)
