from turtle import shapesize
import torch
from torch import nn


def init_weight(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)


def caculate_accuracy(preds: torch.Tensor,
                      targets: torch.Tensor,
                      total_correct: int,
                      total_samples: int):
    correct_nums = (preds.argmax(dim=-1) == targets).sum()
    total_correct += correct_nums.item()
    total_samples += len(preds)
    accuracy = total_correct / total_samples

    return accuracy, total_correct, total_samples


    


