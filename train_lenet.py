import torch
from torch import nn
from models.lenet import LeNet
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from tqdm import tqdm, trange
import pdb


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


if __name__=='__main__':
    # 数据
    train_set = FashionMNIST('./datasets/fashionMNIST',
                            train=True, download=True,
                            transform=ToTensor())
    test_set = FashionMNIST('./datasets/fashionMNIST',
                            train=False, download=True,
                            transform=ToTensor())
    train_loader = DataLoader(dataset=train_set, batch_size=64,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=64,
                              shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 模型
    model = LeNet()
    model.apply(init_weight)
    model = model.to(device)
    # 优化器+损失函数
    optimizer = SGD(model.parameters(), lr=1)
    criteria = CrossEntropyLoss()

    epochs = 10
    for i in trange(epochs, desc='Total Progress'):
        total_loss = 0
        total_correct = 0
        total_samples = 0
        train_loop = tqdm(train_loader, leave=False)
        train_loop.set_description(f'Epoch-train:{i+1}/{epochs}')
        model.train()
        for batch in train_loop:
            img, target = batch
            img = img.to(device)
            target = target.to(device)

            out = model(img)
            loss = criteria(out, target)
            total_loss += loss.item()

            accuracy, total_correct, total_samples = caculate_accuracy(out, target,
                                                                       total_correct, total_samples)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loop.set_postfix(loss=loss.item())
        print(f'train total loss: {total_loss}')
        print(f'accuracy:{accuracy}, total_correct:{total_correct}, total_samples:{total_samples}')

        total_loss = 0
        total_correct = 0
        total_samples = 0
        test_loop = tqdm(test_loader, leave=False)
        test_loop.set_description(f'Epoch-test:{i+1}/{epochs}')
        model.eval()
        with torch.no_grad():
            for batch in test_loop:
                img, target = batch
                img = img.to(device)
                target = target.to(device)

                out = model(img)
                loss = criteria(out, target)
                total_loss += loss.item()
                accuracy, total_correct, total_samples = caculate_accuracy(out, target,
                                                                        total_correct, total_samples)
                test_loop.set_postfix(loss=loss.item())
            print(f'test total loss: {total_loss}')
            print(f'accuracy:{accuracy}, total_correct:{total_correct}, total_samples:{total_samples}')




