import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
from scipy import ndimage
import torch.nn as nn
# Define the CNN model
from torchvision.models import resnet

import torch
import torch.nn as nn

import torch
import torch.nn as nn
from torchvision import models


    
    
class CNN(nn.Module):
    def __init__(self, size=64, num_classes=2,in_slice=16):  # Set default value of size to 64
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_slice, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(32, 32, 2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.layer2 = self._make_layer(64, 64, 2)
        
        # Calculate the size of the flattened feature map after two poolings
        flattened_size = 64 * (size // 4) * (size // 4)
        
        self.fc1 = nn.Linear(flattened_size, 256)
        self.fc2 = nn.Linear(256, out_features=num_classes)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.layer1(x)
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        #x = self.sigmoid(x)
        #x = torch.softmax(x, dim=1)
        return x

    

class SimpleCNN(nn.Module):
    def __init__(self, size=64, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        flattened_size = 32 * (size // 2) * (size // 2)
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        #x = self.sigmoid(x)
        #x = F.softmax(x, dim=1)
        return x
    

import torch
import torch.nn as nn
import torch.optim as optim

class deepCNN(nn.Module):
    def __init__(self, size=64, num_classes=2, in_channels=16):  # Set default value of size to 64
        super(deepCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Calculate the size of the flattened feature map after two poolings
        self.flattened_size = 64 * (size // 4) * (size // 4)

        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x) if self.shortcut else x  # Adjust shortcut connection
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

    
    
class residualCNN(nn.Module):
    def __init__(self, size=128, num_classes=2, in_channels=8):
        super(residualCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            ResidualBlock(16, 16),
            nn.Dropout(0.25),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            ResidualBlock(16, 64),  # input channels of 16, output channels of 64
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer4 = nn.Sequential(
            ResidualBlock(64, 64),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer5 = nn.Sequential(
            ResidualBlock(64, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Calculate the size of the flattened feature map after two poolings
        self.flattened_size = 64 * (size // 4) * (size // 4)

        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x

# Example usage:
# Initialize the model
#model = deepCNN()

# Assuming input size 16x64x64
#input_tensor = torch.randn(1, 16, 64, 64)
#output = model(input_tensor)
#print(output.shape)  # Check the output shape


class EfficientNet(nn.Module):
    def __init__(self, num_classes=2,input_channels=16,model_version='efficientnet_b0',pretrained=True):    
        super(EfficientNet, self).__init__()
        self.model = getattr(models, model_version)(pretrained=pretrained)
        self.model.features[0][0] = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.model(x)






def generate_model(model_type='resnet', model_depth=50,
                   input_W=224, input_H=224, input_D=224, resnet_shortcut='B',
                   no_cuda=False, gpu_id=[0],
                   pretrain_path = 'pretrain/resnet_50.pth',
                   nb_class=1):
    assert model_type in [
        'resnet'
    ]

    if model_type == 'resnet':
        assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = resnet.resnet10(
            sample_input_W=input_W,
            sample_input_H=input_H,
            sample_input_D=input_D,
            shortcut_type=resnet_shortcut,
            no_cuda=no_cuda,
            num_seg_classes=1)
        fc_input = 256
    elif model_depth == 18:
        model = resnet.resnet18(
            sample_input_W=input_W,
            sample_input_H=input_H,
            sample_input_D=input_D,
            shortcut_type=resnet_shortcut,
            no_cuda=no_cuda,
            num_seg_classes=1)
        fc_input = 512
    elif model_depth == 34:
        model = resnet.resnet34(
            sample_input_W=input_W,
            sample_input_H=input_H,
            sample_input_D=input_D,
            shortcut_type=resnet_shortcut,
            no_cuda=no_cuda,
            num_seg_classes=1)
        fc_input = 512
    elif model_depth == 50:
        model = resnet.resnet50(
            sample_input_W=input_W,
            sample_input_H=input_H,
            sample_input_D=input_D,
            shortcut_type=resnet_shortcut,
            no_cuda=no_cuda,
            num_seg_classes=1)
        fc_input = 2048
    elif model_depth == 101:
        model = resnet.resnet101(
            sample_input_W=input_W,
            sample_input_H=input_H,
            sample_input_D=input_D,
            shortcut_type=resnet_shortcut,
            no_cuda=no_cuda,
            num_seg_classes=1)
        fc_input = 2048
    elif model_depth == 152:
        model = resnet.resnet152(
            sample_input_W=input_W,
            sample_input_H=input_H,
            sample_input_D=input_D,
            shortcut_type=resnet_shortcut,
            no_cuda=no_cuda,
            num_seg_classes=1)
        fc_input = 2048
    elif model_depth == 200:
        model = resnet.resnet200(
            sample_input_W=input_W,
            sample_input_H=input_H,
            sample_input_D=input_D,
            shortcut_type=resnet_shortcut,
            no_cuda=no_cuda,
            num_seg_classes=1)
        fc_input = 2048

    model.conv_seg = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten(),
                                   nn.Linear(in_features=fc_input, out_features=nb_class, bias=True))

    if not no_cuda:
        if len(gpu_id) > 1:
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=gpu_id)
            net_dict = model.state_dict()
        else:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id[0])
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=None)
            net_dict = model.state_dict()
    else:
        net_dict = model.state_dict()

    print('loading pretrained model {}'.format(pretrain_path))
    pretrain = torch.load(pretrain_path)
    pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
    # k 是每一层的名称，v是权重数值
    net_dict.update(pretrain_dict) #字典 dict2 的键/值对更新到 dict 里。
    model.load_state_dict(net_dict) #model.load_state_dict()函数把加载的权重复制到模型的权重中去

    print("-------- pre-train model load successfully --------")

    return model






if __name__ == '__main__':
    a = torch.rand(4, 16, 64, 64)
    #print(a)
    model = SimpleCNN()
    logits = model(a)

    # Apply softmax to obtain probabilities
    probabilities = F.softmax(logits, dim=1)
    print(probabilities)
    print(torch.max(probabilities, dim=1))
    # Assuming you have true labels for your inputs
    # Let's create some dummy true labels for demonstration
    true_labels = torch.LongTensor([0, 1, 0, 1])  # Binary labels
    model = generate_model(model_type='resnet', model_depth=50,
                input_W=224, input_H=224, input_D=224, resnet_shortcut='B',
                no_cuda=False, gpu_id=[0],
                pretrain_path = './resnet_50_23dataset.pth',
                nb_class=11)