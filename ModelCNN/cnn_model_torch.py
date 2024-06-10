import torch
import torch.nn as nn
import torch.nn.functional as F

def same_padding(kernel_size, stride, dilation=1):
    return ((stride - 1) + dilation * (kernel_size - 1)) // 2

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # conv1
        self.conv1_1 = nn.Conv2d(1, 64, 3, padding=same_padding(3, 1))
        self.conv1_2 = nn.Conv2d(64, 64, 3, stride=2, padding=same_padding(3, 2))
        self.bn1 = nn.BatchNorm2d(64)

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=same_padding(3, 1))
        self.conv2_2 = nn.Conv2d(128, 128, 3, stride=2, padding=same_padding(3, 2))
        self.bn2 = nn.BatchNorm2d(128)

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=same_padding(3, 1))
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=same_padding(3, 1))
        self.conv3_3 = nn.Conv2d(256, 256, 3, stride=2, padding=same_padding(3, 2))
        self.bn3 = nn.BatchNorm2d(256)

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=same_padding(3, 1))
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=same_padding(3, 1))
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=same_padding(3, 1))
        self.bn4 = nn.BatchNorm2d(512)

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=same_padding(3, 1, dilation=2), dilation=2)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=same_padding(3, 1, dilation=2), dilation=2)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=same_padding(3, 1, dilation=2), dilation=2)
        self.bn5 = nn.BatchNorm2d(512)

        # conv6
        self.conv6_1 = nn.Conv2d(512, 512, 3, padding=same_padding(3, 1, dilation=2), dilation=2)
        self.conv6_2 = nn.Conv2d(512, 512, 3, padding=same_padding(3, 1, dilation=2), dilation=2)
        self.conv6_3 = nn.Conv2d(512, 512, 3, padding=same_padding(3, 1, dilation=2), dilation=2)
        self.bn6 = nn.BatchNorm2d(512)

        # conv7
        self.conv7_1 = nn.Conv2d(512, 256, 3, padding=same_padding(3, 1))
        self.conv7_2 = nn.Conv2d(256, 256, 3, padding=same_padding(3, 1))
        self.conv7_3 = nn.Conv2d(256, 256, 3, padding=same_padding(3, 1))
        self.bn7 = nn.BatchNorm2d(256)

        # conv8
        self.deconv8_1 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.conv8_2 = nn.Conv2d(128, 128, 3, padding=same_padding(3, 1))
        self.conv8_3 = nn.Conv2d(128, 128, 3, padding=same_padding(3, 1))
        self.bn8 = nn.BatchNorm2d(128)

        self.prediction = nn.Conv2d(128, 264, 1, padding=same_padding(1, 1))
        self.output_layer = nn.ConvTranspose2d(264, 264, 3, stride=4, padding=0, output_padding=1)
        
    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.bn1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.bn2(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.bn3(x)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.bn4(x)

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.bn5(x)

        x = F.relu(self.conv6_1(x))
        x = F.relu(self.conv6_2(x))
        x = F.relu(self.conv6_3(x))
        x = self.bn6(x)

        x = F.relu(self.conv7_1(x))
        x = F.relu(self.conv7_2(x))
        x = F.relu(self.conv7_3(x))
        x = self.bn7(x)

        x = F.relu(self.deconv8_1(x))
        x = F.relu(self.conv8_2(x))
        x = F.relu(self.conv8_3(x))
        x = self.bn8(x)

        x = self.prediction(x)
        x = self.output_layer(x)
        return x

if __name__ == '__main__':
    model = CNNModel()
    print(model)
