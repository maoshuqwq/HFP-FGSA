import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义原始特征图的尺寸
batch_size, channels, height, width = 4, 3, 64, 64
feature_map = torch.randn(batch_size, channels, height, width)

# 定义原始格子参数
grids = [
    (10, 10, 20, 20),  # 对于第一个通道
    (15, 15, 25, 25),  # 对于第二个通道
    (5, 5, 30, 30)     # 对于第三个通道
]

# 缩小特征图到原来的1/16
# 使用平均池化作为示例
pool = nn.AvgPool2d(2, stride=2)
feature_map_small = pool(pool(feature_map))  # 连续池化两次

# 调整格子参数到新尺寸
grids_small = [(x//4, y//4, w//4, h//4) for x, y, w, h in grids]

# 应用新的格子遮罩
for batch in range(batch_size):
    for channel in range(channels):
        x, y, w, h = grids_small[channel]
        mask = torch.zeros(feature_map_small.shape[2:], dtype=torch.float32)
        for x,y,h,w in grids_small[channel]:
            mask[y:y+h, x:x+w] = 1
        feature_map_small[batch, channel] *= mask
