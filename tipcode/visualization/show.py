# import torch
# import torchvision.transforms.functional as TF
# import matplotlib.pyplot as plt
# from PIL import Image
# import os
# import numpy as np
#
# # 创建保存目录
# save_root = "vit_feature_maps"
# os.makedirs(save_root, exist_ok=True)
#
# # 自动处理并保存
# def save_feature_maps_all_layers(feature_maps, img_size=(224, 224), num_show=4):
#     for layer_idx, fmap in enumerate(feature_maps):
#         fmap = fmap.cpu()
#         # 保存前几个通道
#         for ch in range(num_show):
#             channel_map = fmap[ch]
#             channel_map -= channel_map.min()
#             channel_map /= channel_map.max()
#             print(channel_map.shape)
#             resized = TF.resize(channel_map.unsqueeze(0), img_size)[0]
#             img = Image.fromarray((resized.numpy() * 255).astype(np.uint8))
#             img_path = os.path.join(save_root, f"layer_{layer_idx}_channel_{ch}.png")
#             img.save(img_path)
#
#         # 保存平均图
#         avg_map = fmap.mean(0)
#         avg_map -= avg_map.min()
#         avg_map /= avg_map.max()
#         resized_avg = TF.resize(avg_map.unsqueeze(0), img_size)[0]
#         avg_img = Image.fromarray((resized_avg.numpy() * 255).astype(np.uint8))
#         avg_path = os.path.join(save_root, f"layer_{layer_idx}_avg.png")
#         avg_img.save(avg_path)
#
#         print(f"Layer {layer_idx}: Saved {num_show} channels + average map.")
#
# # 假设你已经有 feature_maps 列表，每层是 [768, 32, 32]
# feature_maps = torch.rand(768,32,32)
# save_feature_maps_all_layers(feature_maps)


import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 假设你得到的特征图是这个
feature_map = torch.randn(768, 32, 32)  # [C, H, W]

# 方法1：可视化前几个通道
def show_feature_channels(fm, save_prefix="feature"):
    fm = fm.cpu()
    num_show = 6  # 展示前6个通道
    for i in range(num_show):
        fmap = fm[i]
        fmap -= fmap.min()
        fmap /= fmap.max()
        fmap = TF.resize(fmap.unsqueeze(0), [512, 512])[0]  # 恢复到原图大小
        img = Image.fromarray((fmap.numpy() * 255).astype(np.uint8))
        img.save(f"{save_prefix}_channel_{i}.png")
        plt.subplot(1, num_show, i+1)
        plt.imshow(img, cmap='viridis')
        plt.axis('off')
    plt.show()

show_feature_channels(feature_map)

import matplotlib.pyplot as plt
import matplotlib.cm as cm


def show_feature_channels_colormap(fm, save_prefix="feature", colormap='jet'):
    fm = fm.cpu()
    num_show = 6
    for i in range(num_show):
        fmap = fm[i]
        fmap -= fmap.min()
        fmap /= fmap.max()
        fmap_resized = TF.resize(fmap.unsqueeze(0), [512, 512])[0]

        # 应用伪彩色映射
        cmap = cm.get_cmap(colormap)
        colored_map = cmap(fmap_resized.numpy())  # 返回 RGBA
        colored_map = (colored_map[:, :, :3] * 255).astype(np.uint8)  # 取 RGB 且转 uint8

        img = Image.fromarray(colored_map)
        img.save(f"{save_prefix}_channel_{i}.png")

        # 同时显示
        plt.subplot(1, num_show, i + 1)
        plt.imshow(img)
        plt.axis('off')
    plt.show()

show_feature_channels_colormap(feature_map)
# 方法2：通道平均合成一张灰度图
# def show_average_feature_map(fm, save_path="feature_avg.png"):
#     fm = fm.cpu()
#     fmap = fm.mean(0)  # [32, 32]
#     fmap -= fmap.min()
#     fmap /= fmap.max()
#     fmap = TF.resize(fmap.unsqueeze(0), [224, 224])[0]
#     img = Image.fromarray((fmap.numpy() * 255).astype(np.uint8))
#     img.save(save_path)
#     plt.imshow(img, cmap='viridis')
#     plt.axis('off')
#     plt.title('Average Feature Map')
#     plt.show()
#
#
# feature_map = torch.randn(768,32,32)
# show_average_feature_map(feature_map)