import torch.optim as optim
import torch
def one_hot_encode(labels, num_classes):
    one_hot = torch.zeros(len(labels), num_classes)
    one_hot.scatter_(1, labels.view(-1, 1), 1)
    return one_hot

def apply_gain_contrast(images, light=[0.7, 1, 1.3], contrasts=[1, 1.3, 1.6]):
    bs, _, H, W = images.shape
    outputs = []

    for li in light:
        for contrast in contrasts:
            transformed = li*((images - images.mean(dim=(2, 3), keepdim=True)) * contrast + images.mean(dim=(2, 3), keepdim=True))
            outputs.append(transformed)

    return torch.cat(outputs, dim=1)



import torch.nn.functional as F

import torch.nn as nn
import torchvision
class UltraAttentionCNN(nn.Module):
    def __init__(self, baseline, in_channels=9):
        super(UltraAttentionCNN, self).__init__()

        self.base_model = baseline

        # 修改第一个卷积层的输入通道数，将3改为16
        self.base_model.conv1 = nn.Conv2d(9, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # 修改最后的全连接层的输出类别数，将1000改为num_classes
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 4)

        self.depthwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, stride=2,
                                    kernel_size= 13, groups=in_channels, padding=13 // 2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.in_channels = in_channels
        self.fc1 = nn.Linear(in_channels, in_channels // 4)  # Reduce dimension
        self.fc2 = nn.Linear(in_channels // 4, in_channels)

    def forward(self, x):
        x = apply_gain_contrast(x)  #bs, 16, H, W
        tmp = x
        mean = x.mean([0, 2, 3], keepdim=True)  # 计算每个通道的均值
        std = x.std([0, 2, 3], keepdim=True)
        x = (x - mean) / std

        x_conv = self.depthwise_conv(x)
        # x_pool = self.global_avg_pool(x_conv).view(x.size(0), -1)
        x_pool = x_conv.std(dim=[2, 3], keepdim=True).view(x.size(0), -1)
        attention = F.relu(self.fc1(x_pool))
        attention = torch.sigmoid(self.fc2(attention))  # Sigmoid to get attention weights in range [0, 1]
        attention = attention.view(x.size(0), self.in_channels, 1, 1)
        out = x * attention  # Broadcast attention weights over the spatial dimensions
        out = self.base_model(out)
        return out, attention, tmp


def Get_BG_ResNet18():
    return UltraAttentionCNN(torchvision.models.resnet18(pretrained=False))



def Get_BG_ResNet32():
    return UltraAttentionCNN(torchvision.models.resnet34(pretrained=False))