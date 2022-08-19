import torch
from torch import nn


def batch_norm(x: torch.Tensor, gamma, beta,
               moving_mean, moving_var,
               eps, momentum):
    """batch norm自己实现版本"""
    if not torch.is_grad_enabled():
        # 推理阶段，使用全局的均值和方差
        x_hat = (x - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        # 训练阶段，使用每个batch计算得到的均值和方差，并统计全局的均值和方差
        assert len(x.shape) in (2, 4)  # 只针对二维卷积以及全连接层
        if len(x.shape) == 2:
            mean = x.mean(dim=0)
            var = ((x - mean) ** 2).mean(dim=0)
        else:
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = ((x - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + eps)
        moving_mean = momentum * moving_mean + (1 - momentum) * mean
        moving_var = momentum * moving_var + (1 - momentum) * var
    y = gamma * x_hat + beta
    return y, moving_mean.data, moving_var.data


class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super().__init__()
        assert num_dims in (2, 4)
        if num_dims == 2:
            shape = (1, num_features)
        elif num_dims == 4:
            shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, x):
        if self.moving_mean.device != x.device:
            self.moving_mean = self.moving_mean.to(x.device)
            self.moving_var = self.moving_var.to(x.device)
        y, self.moving_mean, self.moving_var = batch_norm(x, self.gamma, self.beta,
                                                          self.moving_mean, self.moving_var,
                                                          eps=1e-5, momentum=0.9)
        return y