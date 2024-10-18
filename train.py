import torch
from model import SimpleCNN,CNN,deepCNN,residualCNN
from dataloader import get_loader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from utils import get_acc, AvgrageMeter
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F  # 用于softmax计算
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)



if __name__ == '__main__':
    dataset_now="ipf"
    
    train_loader, test_loader = get_loader(dataset=dataset_now)
    for images, labels in train_loader:
        print("批次图像形状:", images.shape)  # 输出应为 (batch_size, ...)
        print(labels.shape)
        if labels.dim()>=2:
            if labels.shape[1] == 1:
                labels=labels.squeeze(1)  # 移除第一个维度
        print("更新批次标签:", labels)
        break  # 退出循环，只查看第一个批次
    print("dataset done!")

    model = residualCNN(size=256,in_channels=16)
    model = deepCNN(size=256,in_channels=16)
    model.apply(weights_init)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_weights = torch.tensor([1.0, 10.0]).to(device)  # 类别1的权重比类别0大

    print(device)
    model = model.to(device)
    
    print("model done")

    num_epochs = 30 # 将epoch设置为10roc_auc_score


    CrossEntropyLoss = torch.nn.CrossEntropyLoss(weight=class_weights)
    #CrossEntropyLoss = nn.CrossEntropyLoss()
    #optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    
    train_losses = []  # 存储每个epoch训练集的损失
    test_losses = []   # 存储每个epoch测试集的损失
    test_accuracies = []  # 存储每个epoch测试集的准确率
    train_accuracies = [] # 存储每个epoch训练集的准确率
    train_aucs = []  # 存储每个epoch训练集的AUC值
    test_aucs = []   # 存储每个epoch测试集的AUC值
    
    for epoch in range(num_epochs):
        model.train()
        train_loss_avg = AvgrageMeter()  # 用于计算训练集上的平均损失
        train_correct = 0
        train_total = 0
        all_train_labels = []  # 存储训练集的所有标签
        all_train_outputs = []  # 存储训练集的所有模型输出
    
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            if labels.dim() >= 2:
                if labels.shape[1] == 1:
                    labels = labels.squeeze(1)  # 移除第一个维度
            optimizer.zero_grad()
            outputs = model(images)
            loss = CrossEntropyLoss(outputs, labels)
            train_loss_avg.update(loss.item(), images.size(0))  # 更新平均损失
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
    
            # 将标签和softmax后的输出保存下来，用于计算AUC
            all_train_labels.append(labels.cpu().numpy())
            all_train_outputs.append(F.softmax(outputs, dim=1)[:, 1].cpu().detach().numpy())
    
            loss.backward()
            optimizer.step()
    
        train_losses.append(train_loss_avg.avg)  # 将每个epoch的平均损失保存起来
        train_accuracy = train_correct / train_total
        train_accuracies.append(train_accuracy)  # 将每个epoch的训练集准确率保存起来
    
        # 计算训练集AUC
        all_train_labels = np.concatenate(all_train_labels)
        all_train_outputs = np.concatenate(all_train_outputs)
        print(all_train_outputs)
        train_auc = roc_auc_score(all_train_labels, all_train_outputs)  # 适用于多分类

        train_aucs.append(train_auc)
    
        model.eval()
        test_loss_avg = AvgrageMeter()  # 用于计算测试集上的平均损失
        test_correct = 0
        test_total = 0
        all_test_labels = []  # 存储测试集的所有标签
        all_test_outputs = []  # 存储测试集的所有模型输出
    
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                images, labels = images.to(device), labels.to(device)
                if labels.dim() >= 2:
                    if labels.shape[1] == 1:
                        labels = labels.squeeze(1)  # 移除第一个维度
                outputs = model(images)
                loss = CrossEntropyLoss(outputs, labels)
                test_loss_avg.update(loss.item(), images.size(0))  # 更新平均损失
                correct, total = get_acc(outputs, labels)
                test_correct += correct
                test_total += total
                _, predicted = torch.max(outputs.data, 1)
    
                # 将标签和softmax后的输出保存下来，用于计算AUC
                all_test_labels.append(labels.cpu().numpy())
                all_test_outputs.append(F.softmax(outputs, dim=1)[:, 1].cpu().detach().numpy())
    
            test_losses.append(test_loss_avg.avg)  # 将每个epoch的平均损失保存起来
            test_accuracy = test_correct / test_total
            test_accuracies.append(test_accuracy)
    
            # 计算测试集AUC
            all_test_labels = np.concatenate(all_test_labels)
            all_test_outputs = np.concatenate(all_test_outputs)
            print(all_test_outputs)
            test_auc = roc_auc_score(all_test_labels, all_test_outputs)  # 适用于多分类

            test_aucs.append(test_auc)
    
        # 输出每个epoch的损失、准确率和AUC
        print("Epoch [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.4f}, Train AUC: {:.4f}, Test Loss: {:.4f}, Test Accuracy: {:.4f}, Test AUC: {:.4f}"
              .format(epoch+1, num_epochs, train_loss_avg.avg, train_accuracy, train_auc, test_loss_avg.avg, test_accuracy, test_auc))
    

    # 绘制损失曲线
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig('/home/featurize/for2d_all/figure/losscurve.png')
    plt.show()

    # 绘制准确率曲线
    plt.plot(range(1, num_epochs+1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, num_epochs+1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.savefig('/home/featurize/for2d_all/figure/accuractcurve.png')
    plt.show()
