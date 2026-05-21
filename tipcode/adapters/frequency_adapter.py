import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from segment_anything.modeling import Sam
from safetensors import safe_open
from safetensors.torch import save_file

from icecream import ic
import torch.fft


#b,32,32,768
#test input
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


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

#show_feature_channels_colormap(feature_map)

class senet(nn.Module):
    def __init__(self,c=768,r=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(c,c//r,1,1,0,bias=True),nn.ReLU(),nn.Conv2d(c//r,c,1,1,0,bias=True))
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m,nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias,std=1e-6)
        self.apply(_init_weights)

    def forward(self,x):
        res = x
        b,c,h,w=x.size()
        #x = x.view(b,c,h*w)
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out+max_out
        x = x*self.sigmoid(out)
        #x = x.view(b,c,h,w)
        return x+res

class QuickGELU(nn.Module):
    def forward(self,x:torch.Tensor):
        return x*torch.sigmoid(1.702*x)
class fre_adapter(nn.Module):
    def __init__(self,c=768,r=12):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(c,c//r,bias=True),QuickGELU(),nn.Linear(c//r,c,bias=True))
        self.fc2 = nn.Sequential(nn.Linear(c,c//r,bias=True),QuickGELU(),nn.Linear(c//r,c,bias=True))
        self.IN = nn.LayerNorm(c)
        self.init_weights()
        #self.reduce = nn.Sequential(nn.Linear(2*c,c),QuickGELU())
        #self.se = senet()

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias,std=1e-6) 
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias,std=1e-6)
        self.apply(_init_weights)
    
    def forward(self,x,fre_mask):
        ori = x
        #print(fre_mask.size())
        # 设备无关：跟随输入特征所在设备（cuda/cpu/mps）
        fre_mask_expand = fre_mask.unsqueeze(1).expand(-1,768,-1,-1).to(x.device)
        x_change = x.permute(0,3,1,2)
        mask_x = x_change * fre_mask_expand
        #mask_x = mask_x[0]
        #show_feature_channels_colormap(mask_x)
        x_fre = mask_x.permute(0,2,3,1) 
        #print(x.size())
        #x_fft = torch.fft.fft2(x)#.real#.astype(float)
        #x_fft = x_fft.real.float()
        #print(x_fft.size())
        b,h,w,c = x.size()

        out1 = self.fc1(self.IN(x.view(b,h*w,c))).view(b,h,w,c)
        out2 = self.fc2(self.IN(x_fre.view(b,h*w,c))).view(b,h,w,c)
        #.real
        #real_part = x_fft.reshape(b,h*w, c)
        #print(real_part.size())
        #processed_real = self.linear(real_part)
        #processed_real = self.fc1(self.IN(real_part))
        # 假设处理后的实部大小与原始实部相同
        #processed_fft = processed_real.view(x_fft.shape)
        #processed_fft = torch.complex(processed_real.view(x_fft.shape), torch.zeros_like(processed_real).view(x_fft.shape))
        
        # 步骤 3: 频域回到空域
        #x_ifft = torch.fft.ifft2(processed_fft)
        #x_ifft = x_ifft.real.float()
        #out = self.fc(x_fft.view(b,h*w,c))
        #out2 = x_ifft.view(b,h,w,c)
        return out1+out2+ori


class ChannelPool(nn.Module):
    """通道池化：平均池化 + 最大池化，压缩为单通道"""
    def __init__(self, pool_types=['avg', 'max']):
        super().__init__()
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = torch.mean(x, 1).unsqueeze(1)
                channel_att_raw = avg_pool
            elif pool_type == 'max':
                max_pool = torch.max(x, 1)[0].unsqueeze(1)
                channel_att_raw = max_pool
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        return channel_att_sum


class FGSAttn_Adapter(nn.Module):
    """
    基于FGSAttn思想的可学习频率适配器，替代硬选框FGA。
    在ViT每层后注入频率信息，内部做FFT频率分组加权，完全可学习。
    
    输入: ViT特征 [B, H, W, C]
    输出: 增强特征 [B, H, W, C], 频率注意力图 [B, H, W]
    """
    def __init__(self, dim=768, double_R=64, d=1, group=1, pool_types=['avg', 'max'], init_values=0.):
        super().__init__()
        self.compress = ChannelPool(pool_types)
        self.group = group
        self.fre_interval = d
        self.dim = dim
        self.double_R = double_R
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        
        # 频率组数 = int(R/d)+1, 其中 R = double_R/2
        R = double_R // 2
        self.num_groups = int(R / d) + 1
        
        # 每组独立的可学习FC层（非共享）
        self.fc = nn.Sequential(*[nn.Sequential(
            nn.Linear(self.num_groups, self.num_groups, bias=True),
            nn.LeakyReLU()
        ) for _ in range(group)])
        
        # LayerNorm用于特征归一化
        self.norm = nn.LayerNorm(dim)
        
        # 初始化
        self._init_weights()
        
        # 预计算频率分组掩码（注册为buffer，不参与训练）
        self._build_and_register_freq_mask(64, 64)  # 默认64x64特征图

    def _init_weights(self):
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

    def _build_and_register_freq_mask(self, H, W):
        """预计算频率分组掩码并注册为buffer"""
        center_h, center_w = H // 2, W // 2
        R = min(center_h, center_w)
        d = self.fre_interval
        
        y_coords = torch.arange(H).float()
        x_coords = torch.arange(W).float()
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        r = torch.sqrt((yy - center_h) ** 2 + (xx - center_w) ** 2)
        r = torch.clamp(r, max=R)
        mask = torch.ceil(r / d).long()
        
        self.register_buffer('freq_mask', mask)

    def _min_max(self, x, to_min, to_max):
        """Min-Max归一化到[to_min, to_max]"""
        x_min = torch.min(x)
        x_max = torch.max(x)
        return to_min + ((to_max - to_min) / (x_max - x_min + 1e-8)) * (x - x_min)

    def forward(self, x):
        """
        前向传播
        Args:
            x: ViT特征 [B, H, W, C]
        Returns:
            out: 增强特征 [B, H, W, C]
            attention_maps: 频率注意力图 [B, H, W]
        """
        B, H, W, C = x.shape
        residual = x
        
        # 如果特征图尺寸与预计算掩码不匹配，重新构建
        if self.freq_mask.shape[0] != H or self.freq_mask.shape[1] != W:
            self._build_and_register_freq_mask(H, W)
        
        # 转为 [B, C, H, W] 做通道压缩
        x_conv = x.permute(0, 3, 1, 2)
        
        attention_maps = []
        new_features = []
        
        for i in range(self.group):
            # 通道分组
            K = C // self.group
            feature_i = x_conv[:, i*K:(i+1)*K, :, :]  # [B, K, H, W]
            
            # 通道压缩：平均池化 + 最大池化
            feature_compress = self.compress(feature_i).squeeze(1)  # [B, H, W]
            
            batch_features = []
            batch_attn_maps = []
            
            for b in range(B):
                feature_map = feature_compress[b]  # [H, W]
                
                # FFT变换
                f = torch.fft.fft2(feature_map)
                shift2center = torch.fft.fftshift(f)
                amplitude = torch.abs(shift2center)
                phase = torch.angle(shift2center)
                
                # 频率分组加权
                mask = self.freq_mask  # [H, W]
                mask_labels = mask.unique()
                
                # 每组计算平均振幅
                fre_avg_pool = []
                for label in mask_labels:
                    group_mean = amplitude[mask == label].mean()
                    fre_avg_pool.append(group_mean)
                fre_avg_pool = torch.tensor(fre_avg_pool, device=x.device, dtype=torch.float32)
                
                # 可学习FC加权
                fre_att = self.fc[i](fre_avg_pool)
                
                # 振幅调制
                amplitude_modulated = amplitude.clone()
                for idx in range(len(fre_att)):
                    label = mask_labels[idx]
                    attn = fre_att[idx]
                    amplitude_modulated[mask == label] = (amplitude * attn)[mask == label]
                
                # 逆变换重建
                merged = torch.multiply(torch.exp(1j * phase), amplitude_modulated)
                out = torch.fft.ifftshift(merged)
                out = torch.fft.ifft2(out)
                new_feature_map = torch.real(out)
                
                # 生成注意力图 [H, W]
                attn_map = self._min_max(new_feature_map, 0, 1)
                batch_attn_maps.append(attn_map)
                
                # 特征增强：原始特征 * 注意力图
                enhanced = feature_i[b] * attn_map.unsqueeze(0)  # [K, H, W]
                batch_features.append(enhanced)
            
            # 拼接batch
            batch_features = torch.stack(batch_features)  # [B, K, H, W]
            batch_attn_maps = torch.stack(batch_attn_maps)  # [B, H, W]
            
            new_features.append(batch_features)
            attention_maps.append(batch_attn_maps)
        
        # 合并所有group的特征
        new_features = torch.cat(new_features, dim=1)  # [B, C, H, W]
        new_features = new_features.permute(0, 2, 3, 1)  # [B, H, W, C]
        
        # 取所有group注意力图的平均
        attention_maps = torch.stack(attention_maps).mean(dim=0)  # [B, H, W]
        
        # 残差融合
        out = residual + self.gamma * new_features
        
        return out, attention_maps
