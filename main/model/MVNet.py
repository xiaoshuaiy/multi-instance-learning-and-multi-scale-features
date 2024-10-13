import torch.nn as nn
import torch
class Block3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1), padding=0):
        super(Block3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batchnorm = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x

class Block2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Block2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x

class MVNet(nn.Module):
    def __init__(self):
        super(MVNet, self).__init__()

        # Anisotropic probing layers
        self.block3d_1 = Block3D(1, 32, (5, 1, 1),(5, 1, 1))
        self.block3d_2 = Block3D(32, 64, (3, 1, 1),(3, 1, 1))
        self.block3d_3 = Block3D(64, 1, (2, 1, 1),)

        # 2D NIN
        self.block2d_1 = Block2D(1, 32, 4)
        self.block2d_2 = Block2D(32, 64, 3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2,ceil_mode=True)
        self.dropout1 = nn.Dropout(0.2)

        self.block2d_3 = Block2D(64, 64,3)
        self.block2d_4 = Block2D(64, 128, 3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2,ceil_mode=True)
        self.dropout2 = nn.Dropout(0.2)

        self.block2d_5 = Block2D(128, 128, 3)
        self.block2d_6 = Block2D(128, 256, 3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.dropout3 = nn.Dropout(0.2)

        self.view = nn.Flatten()

    def forward(self, Mvx):
        feature_MV=[]
        for i in range(3):
            if i==1:
                x = Mvx.permute(0,1,3,2,4)
            if i==2:
                x = Mvx.permute(0,1,4,2,3)
            else:
                x = Mvx
            x = self.block3d_1(x)
            x = self.block3d_2(x)
            x = self.block3d_3(x)
            x = x.squeeze(dim=2)

            x = self.block2d_1(x)
            x = self.block2d_2(x)
            x = self.pool1(x)
            x = self.dropout1(x)

            x = self.block2d_3(x)
            x = self.block2d_4(x)
            x = self.pool2(x)
            x = self.dropout2(x)

            x = self.block2d_5(x)
            x = self.block2d_6(x)
            x = self.pool3(x)
            x = self.dropout3(x)

            # x = self.pool3(x)
            x = self.view(x)
            feature_MV.append(x)
        features = torch.cat((feature_MV[0],feature_MV[1],feature_MV[2]),dim=1)
        return features