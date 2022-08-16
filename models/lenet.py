import torch
from torch import nn


class LeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.AvgPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(16*5*5, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )
    
    def forward(self, x):
        out = self.net(x)

        return out


if __name__ == '__main__':
    x = torch.randn(1, 1, 28, 28)
    model = LeNet()
    for layer in model.net:
        x = layer(x)
        print(f'{layer} ouput_shape:{x.shape}')