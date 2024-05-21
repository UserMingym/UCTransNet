import numpy as np
import torch
from torch import nn
from torch.nn import init

class SEAttention(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化层，将每个图像的空间维度压缩为1x1
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # 缩减，减少参数数量和计算量
            nn.ReLU(inplace=True),  # 激活层，非线性
            nn.Linear(channel // reduction, channel, bias=False),  # 升维，恢复到原通道数
            nn.Sigmoid()  # Sigmoid激活函数，将输出值映射到0到1之间
        )

    # 权重初始化方法
    def init_weights(self):
        for m in self.modules():  # 遍历模块中的所有子模块
            if isinstance(m, nn.Conv2d):  # 对于卷积层
                init.kaiming_normal_(m.weight, mode='fan_out')  # 使用kaiming初始化方法初始化权重
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 如果有偏置项，初始化为0
            elif isinstance(m, nn.BatchNorm2d):  # 对于批归一化层
                init.constant_(m.weight, 1)  # 权重初始化为1
                init.constant_(m.bias, 0)  # 偏置初始化为0
            elif isinstance(m, nn.Linear):  # 对于全连接层
                init.normal_(m.weight, std=0.001)  # 权重使用正态分布初始化
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 偏置初始化为0

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

if __name__ == '__main__':
    input = torch.randn(50, 512, 7, 7)
    se = SEAttention(channel=512, reduction=8)
    output = se(input)
    print(output.shape)
