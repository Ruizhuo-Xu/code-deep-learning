from turtle import forward
import torch
from torch import nn
from torchstat import stat


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, strides=1, use_1x1conv=False):
        """
        残差块:
        一般情况下in_channels == out_channels, strides=1, use_1x1conv=False
        或者, 2 * in_channels == out_channels, strides=2, use_1x1conv=True
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return self.relu(Y)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_residuals, first_block=False):
        super().__init__()
        self._blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                self._blk.append(Residual(in_channels, out_channels, strides=2, use_1x1conv=True))
            else:
                self._blk.append(Residual(out_channels, out_channels, strides=1, use_1x1conv=False))
        self.blk = nn.Sequential(*self._blk)

    def forward(self, X):
        return self.blk(X)
        


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.blk0 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.blk1 = ResidualBlock(64, 64, 2, True)
        self.blk2 = ResidualBlock(64, 128, 2)
        self.blk3 = ResidualBlock(128, 256, 2)
        self.blk4 = ResidualBlock(256, 512, 2)
        self.net = nn.Sequential(
            self.blk0,
            self.blk1,
            self.blk2,
            self.blk3,
            self.blk4,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 10),
        )

    def forward(self, X):
        return self.net(X)


if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    model = ResNet18()
    # for layer in model.net:
    #     x = layer(x)
    #     print(f'{layer} ouput_shape:{x.shape}')
    stat(model, (1, 224, 224))