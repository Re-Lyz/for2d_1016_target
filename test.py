import torch
from model import SimpleCNN
from model import generate_model
from dataloader import get_loader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from utils import get_acc, AvgrageMeter
import matplotlib.pyplot as plt

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


if __name__ == '__main__':
    train_loader, test_loader = get_loader()
    for images, labels in train_loader:
        print("批次图像形状:", images.shape)  # 输出应为 (batch_size, ...)
        print("批次标签:", labels)
        break  # 退出循环，只查看第一个批次
    print("dataset done!")

    model = generate_model(model_type='resnet', model_depth=50,
                   input_W=224, input_H=224, input_D=224, resnet_shortcut='B',
                   no_cuda=False, gpu_id=[0],
                   pretrain_path = './resnet_50_23dataset.pth',
                   nb_class=11)

    