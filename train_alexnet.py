import torch
from torch import nn
from models.alexnet import AlexNet
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor, Resize, Compose
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from tqdm import tqdm, trange
import pdb
import wandb
import time


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
    config = {'epochs': 10, 'lr': 0.1,
              'model': 'AlexNet',
              'batch_size': 32}
    wandb.init(project='fashionMNIST-train',
               name=f'AlexNet-train',
               config=config)
    # 数据
    transforms = Compose([ToTensor(), Resize(size=(224, 224))])
    train_set = FashionMNIST('./datasets/fashionMNIST',
                            train=True, download=True,
                            transform=transforms)
    test_set = FashionMNIST('./datasets/fashionMNIST',
                            train=False, download=True,
                            transform=transforms)
    train_loader = DataLoader(dataset=train_set, batch_size=config['batch_size'],
                              shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=config['batch_size'],
                              shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 模型
    model = AlexNet()
    model.apply(init_weight)
    model = model.to(device)
    # 优化器+损失函数
    optimizer = SGD(model.parameters(), lr=config['lr'])
    criteria = CrossEntropyLoss()

    for i in trange(config['epochs'], desc='Total Progress'):
        train_total_loss = 0
        total_correct = 0
        total_samples = 0
        train_loop = tqdm(train_loader, leave=False)
        train_loop.set_description(f'Epoch-train:{i+1}/{config["epochs"]}')
        model.train()
        start = time.time()
        for batch in train_loop:
            img, target = batch
            img = img.to(device)
            target = target.to(device)

            out = model(img)
            loss = criteria(out, target)
            train_total_loss += loss.item()

            train_accuracy, total_correct, total_samples = caculate_accuracy(out, target,
                                                                       total_correct, total_samples)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loop.set_postfix(loss=loss.item())
        end = time.time()
        train_loss = train_total_loss / len(train_loop)
        print(f'train_loss: {train_loss}')
        print(f'accuracy:{train_accuracy}, total_correct:{total_correct}, total_samples:{total_samples}')

        test_total_loss = 0
        total_correct = 0
        total_samples = 0
        test_loop = tqdm(test_loader, leave=False)
        test_loop.set_description(f'Epoch-test:{i+1}/{config["epochs"]}')
        model.eval()
        with torch.no_grad():
            for batch in test_loop:
                img, target = batch
                img = img.to(device)
                target = target.to(device)

                out = model(img)
                loss = criteria(out, target)
                test_total_loss += loss.item()
                test_accuracy, total_correct, total_samples = caculate_accuracy(out, target,
                                                                           total_correct, total_samples)
                test_loop.set_postfix(loss=loss.item())
            test_loss = test_total_loss / len(test_loop)
            print(f'test_loss: {test_loss}')
            print(f'accuracy:{test_accuracy}, total_correct:{total_correct}, total_samples:{total_samples}')
            wandb.log({
                'test/loss': test_loss,
                'test/accuracy': test_accuracy,
                'train/loss': train_loss,
                'train/accuracy': train_accuracy,
                'train/time_per_epoch': end - start,
            })