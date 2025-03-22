import os
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch.optim as optim
import torch
def one_hot_encode(labels, num_classes):
    one_hot = torch.zeros(len(labels), num_classes)
    one_hot.scatter_(1, labels.view(-1, 1), 1)
    return one_hot
from function import setup_seed
setup_seed(20)

# 数据目录
data_dir = r'./新胎儿数据集/train'

# 数据预处理：调整图像大小，转换为灰度图，转为tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为224x224
    transforms.Grayscale(num_output_channels=1),  # 转为灰度图
    transforms.ToTensor(),  # 转为Tensor，并归一化到[0, 1]
])

# 使用ImageFolder读取数据，注意要将训练集的子文件夹作为分类标签
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# 按照7:3划分训练集和测试集
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 创建训练和测试集的DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 输出数据集信息
dataset_sizes = {'train': len(train_dataset), 'test': len(test_dataset)}
print(f"训练集样本数: {dataset_sizes['train']}")
print(f"测试集样本数: {dataset_sizes['test']}")

# 获取类别名称
class_names = dataset.classes
for index, class_name in enumerate(class_names):
    print(f"Label: {index}, Class Name: {class_name}")




setup_seed(50)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from Oures import Get_BG_ResNet18, Get_BG_ResNet32

model = Get_BG_ResNet18().to(device)
# model = Get_BG_ResNet32().to(device)

total_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"模型的总参数数量: {total_params:.2f}M")


import time

# # 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
Max = 0
for i in range(20):
    model.train()
    p = 0
    sum_loss = torch.zeros((30))
    for inputs, labels in train_loader:
        labels = one_hot_encode(labels, 4)
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss[p] = loss
        p = p + 1
    print(torch.mean(sum_loss))
    if i>=4:
        start_time = time.time()
        model.eval()
        all_test_target = []
        all_test_output = []
        all_score = []
        m = 0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            all_test_target.append(labels)
            output = model(inputs)
            all_score.append(output)
            predicted_class = torch.argmax(output, dim=1).to(device)
            all_test_output.append(predicted_class)
            m = m + 1
        end_time = time.time()
        elapsed_time = end_time - start_time  # 计算执行时间
        print(f"代码执行时间: {elapsed_time:.4f} 秒")

        all_test_target = torch.cat(all_test_target)
        all_test_output = torch.cat(all_test_output)
        all_score = torch.cat(all_score)
        acu = torch.sum(all_test_output == all_test_target).item() / len(all_test_output)
        acu_percent = acu * 100
        all_test_target = all_test_target.cpu().numpy()
        all_test_output = all_test_output.cpu().numpy()
        print(f'Accuracy: {acu_percent:.2f}%')






import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 1. 加载预训练的 ResNet 模型
model.eval()
model = model.to("cpu")
# 2. 选择目标卷积层（ResNet 的 layer4 末端）
target_layer = model.base_model.layer4[-1].to("cpu")

# 3. 读取和预处理图片
image_path = "1.jpg"  # 替换成你的图片路径
image = Image.open(image_path)
input_tensor = transform(image).unsqueeze(0).to("cpu")  # 增加 batch 维度

# 4. 计算 Grad-CAM
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 注册前向传播 hook
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[:, class_idx]
        score.backward()

        # 计算 Grad-CAM
        gradients = self.gradients.mean(dim=[2, 3], keepdim=True)  # 全局平均池化
        cam = (self.activations * gradients).sum(dim=1, keepdim=True)  # 加权求和
        cam = F.relu(cam)  # 只保留正相关部分
        cam = cam.squeeze().detach().numpy()
        cam = cv2.resize(cam, (224, 224))  # 调整到原图大小
        cam = (cam - cam.min()) / (cam.max() - cam.min())  # 归一化
        return cam

# 5. 计算类别预测并生成 Grad-CAM
output = model(input_tensor)
pred_class = output.argmax().item()
grad_cam = GradCAM(model, target_layer)
heatmap = grad_cam.generate_cam(pred_class)

# 6. 可视化结果
def show_cam_on_image(image_path, heatmap):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

show_cam_on_image(image_path, heatmap)
