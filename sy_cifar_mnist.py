import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import sys
sys.path.append('/cpfs04/user/hanyujin/rule-gen/rule_tokenizer')
from torchvision import transforms
from utils.data import random_crop_arr, center_crop_arr
# # 设备配置
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")

# # 数据预处理（添加数据增强）
# transform_train = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomCrop(32, padding=4),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
# ])

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
# ])

# # 加载数据集
# # train_dataset = torchvision.datasets.CIFAR10(
# #     root='/cpfs04/user/hanyujin/rule-gen/datasets',
# #     train=True,
# #     download=False,
# #     transform=transform_train
# # )

# # test_dataset = torchvision.datasets.CIFAR10(
# #     root='/cpfs04/user/hanyujin/rule-gen/datasets',
# #     train=False,
# #     download=False,
# #     transform=transform_test
# # )

# train_dataset = torchvision.datasets.MNIST(
#     root='/cpfs04/user/hanyujin/rule-gen/datasets',
#     train=True,
#     download=False,
#     transform=transform_train
# )

# test_dataset = torchvision.datasets.MNIST(
#     root='/cpfs04/user/hanyujin/rule-gen/datasets',
#     train=False,
#     download=False,
#     transform=transform_test
# )

# # 创建数据加载器
# batch_size = 256
# train_loader = torch.utils.data.DataLoader(
#     train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
# test_loader = torch.utils.data.DataLoader(
#     test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# # 定义CNN模型
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.features = nn.Sequential(
#             # 输入: 3x32x32
#             nn.Conv2d(3, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 64x16x16
            
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 128x8x8
            
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 256x4x4
#         )
        
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(256 * 4 * 4, 512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.3),
#             nn.Linear(512, 10)
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x

# model = CNN().to(device)

# # 定义损失函数和优化器
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5, verbose=True)

# # 创建保存目录
# save_dir = "/cpfs04/user/hanyujin/rule-gen/model_cpkt"
# os.makedirs(save_dir, exist_ok=True)
# best_acc = 0.0

# # 评估函数
# def evaluate(loader):
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for images, labels in loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     return 100 * correct / total

# # 训练循环
# epoch_num = 500
# for epoch in range(epoch_num):
#     model.train()
#     running_loss = 0.0
#     for i, (images, labels) in enumerate(train_loader):
#         images, labels = images.to(device), labels.to(device)
        
#         # 前向传播
#         outputs = model(images)
#         loss = criterion(outputs, labels)
        
#         # 反向传播
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         running_loss += loss.item()

#     # 每10个epoch评估
#     if (epoch + 1) % 10 == 0 or epoch == 0:
#         train_acc = evaluate(train_loader)
#         test_acc = evaluate(test_loader)
#         scheduler.step(test_acc)  # 学习率调度
        
#         # 保存最佳模型
#         if test_acc > best_acc:
#             best_acc = test_acc
#             torch.save({
#                 'epoch': epoch+1,
#                 'state_dict': model.state_dict(),
#                 'best_acc': best_acc,
#                 'optimizer' : optimizer.state_dict(),
#             }, os.path.join(save_dir, "mnist_class_cnn_best.pth"))
        
#         print(f"Epoch [{epoch+1}/{epoch_num}], "
#               f"Loss: {running_loss/len(train_loader):.4f}, "
#               f"Train Acc: {train_acc:.2f}%, "
#               f"Test Acc: {test_acc:.2f}%")

# print(f"Best Test Accuracy: {best_acc:.2f}%")

# # Best Test Accuracy: 90.33%


# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
# import torch
# import numpy as np
# from tqdm import tqdm
# from torchvision.datasets import ImageFolder
# from torch.utils.data import DataLoader
# import os
# import sys
# sys.path.append('/cpfs04/user/hanyujin/rule-gen/rule_tokenizer')
# from torchvision import transforms
# from utils.data import random_crop_arr, center_crop_arr

# # 设备配置
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")

# # 数据预处理（适配MNIST特性）
# # transform_train = transforms.Compose([
# #     transforms.RandomRotation(10),       # MNIST适用旋转增强
# #     transforms.RandomAffine(0, shear=10),# 剪切变换
# #     transforms.ToTensor(),
# #     transforms.Normalize((0.1307,), (0.3081,))  # MNIST专用归一化参数
# # ])

# transform_train = transforms.Compose([
#     transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, 64)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
# ])

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# # 加载MNIST数据集
# train_dataset = torchvision.datasets.MNIST(
#     root='/cpfs04/user/hanyujin/rule-gen/datasets',
#     train=True,
#     download=False,
#     transform=transform_train
# )

# test_dataset = torchvision.datasets.MNIST(
#     root='/cpfs04/user/hanyujin/rule-gen/datasets',
#     train=False,
#     download=False,
#     transform=transform_test
# )

# # 创建数据加载器（调整批次大小）
# batch_size = 512
# train_loader = torch.utils.data.DataLoader(
#     train_dataset, batch_size=batch_size, shuffle=True, 
#     num_workers=4, pin_memory=True, persistent_workers=True)

# test_loader = torch.utils.data.DataLoader(
#     test_dataset, batch_size=batch_size, shuffle=False,
#     num_workers=4, pin_memory=True, persistent_workers=True)

# # 定义适配MNIST的CNN模型
# class MNIST_CNN(nn.Module):
#     def __init__(self):
#         super(MNIST_CNN, self).__init__()
#         self.features = nn.Sequential(
#             # 输入: 1x28x28
#             nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 输入通道改为1
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),  # 输出: 32x14x14
            
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),  # 输出: 64x7x7
            
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d(1)  # 全局平均池化输出: 128x1x1
#         )
        
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(128, 10)
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x

# model = MNIST_CNN().to(device)

# # 定义优化策略
# num_epoch = 500
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
# scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, 
#                                         steps_per_epoch=len(train_loader), 
#                                         epochs=num_epoch)

# # 创建保存目录
# save_dir = "/cpfs04/user/hanyujin/rule-gen/model_cpkt"
# os.makedirs(save_dir, exist_ok=True)
# best_acc = 0.0

# # 评估函数（保持不变）
# def evaluate(loader):
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for images, labels in loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     return 100 * correct / total

# # 训练循环（调整学习率调度）
# # num_epoch = 500

# for epoch in range(num_epoch):
#     model.train()
#     running_loss = 0.0
    
#     for i, (images, labels) in enumerate(train_loader):
#         images, labels = images.to(device), labels.to(device)
        
#         # 前向传播
#         outputs = model(images)
#         loss = criterion(outputs, labels)
        
#         # 反向传播
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         scheduler.step()  # 更新学习率
        
#         running_loss += loss.item()

#     # 每10个epoch评估
#     if (epoch + 1) % 10 == 0 or epoch == 0:
#         train_acc = evaluate(train_loader)
#         test_acc = evaluate(test_loader)
        
#         # 保存最佳模型
#         if test_acc > best_acc:
#             best_acc = test_acc
#             torch.save({
#                 'epoch': epoch+1,
#                 'state_dict': model.state_dict(),
#                 'best_acc': best_acc,
#                 'optimizer' : optimizer.state_dict(),
#                 'scheduler' : scheduler.state_dict()
#             }, os.path.join(save_dir, "mnist_cnn_best.pth"))
        
#         print(f"Epoch [{epoch+1}/500], "
#               f"Loss: {running_loss/len(train_loader):.4f}, "
#               f"Train Acc: {train_acc:.2f}%, "
#               f"Test Acc: {test_acc:.2f}%")

# print(f"Best Test Accuracy: {best_acc:.2f}%")