import torch

import torch.nn as nn
import torch.nn.functional as F

def frequency_mask(y_indices,x_indices,h,w):

    grids_small = [(y//16, x//16, h//16, w//16) for y,x in zip(y_indices,x_indices)]

    mask = torch.zeros((32,32), dtype=torch.float32)
    for y,x,h,w in grids_small:
        mask[y:y+h, x:x+w] = 1
    #feature_map_small[batch, channel] *= mask
    #print(mask.size())
    return mask    

def sliding_window_sum(tensor, window_size):
    # 使用unfold方法进行滑窗操作
    """
    对输入张量进行滑窗操作，计算每个滑窗的和
    参数:
        tensor (torch.Tensor): 输入一维张量，形状为 [H, W]
        window_size (int): 滑窗大小
    返回:
        sums (torch.Tensor): 每个滑窗的和，形状为 [H, W, window_size, window_size]
    具体来说，假设输入的张量为 [64, 64],滑窗大小为32,步长同样为32,
    那么unfolded的形状为 [2, 2, 32, 32]
    sums的形状为 [2, 2],sum中的每个元素为4个窗口的所有元素的和
    """
    unfolded = tensor.unfold(0, window_size, window_size).unfold(1, window_size, window_size)
    # 计算每个滑窗的和
    sums = unfolded.sum(dim=(-1, -2))
    return sums

def top_k_windows(sum_tensor, k,window_size):
    # 展平张量，取topk最大值的索引
    k_max_values, k_max_indices = torch.topk(sum_tensor.contiguous().view(-1), k)
    # 一维索引 → 二维坐标（行y，列x）
    y_indices, x_indices = torch.div(k_max_indices, sum_tensor.shape[1], rounding_mode='floor'), k_max_indices % sum_tensor.shape[1]
    # 乘以窗口大小，还原到原始特征图的坐标
    return y_indices*window_size, x_indices*window_size

def get_values_coordinates_and_threshold(tensor, y_indices, x_indices, n, window_size):
    coordinates = {'max': [], 'min': []}
    thresholded_values = {'max': [], 'min': []}
    for y, x in zip(y_indices, x_indices):
        window = tensor[ y:y+window_size, x:x+window_size]
        max_values, max_indices = torch.topk(window.contiguous().view(-1), n)
        min_values, min_indices = torch.topk(window.contiguous().view(-1), n, largest=False)

        # 将线性索引转换为2D坐标，并检查值是否大于0.5
        for idx, val in zip(max_indices, max_values):
            dy, dx = idx.item() // window_size, idx.item() % window_size
            coordinates['max'].append((y + dy, x + dx))
            thresholded_values['max'].append(1 if val > 0.5 else 0)

        for idx, val in zip(min_indices, min_values):
            dy, dx = idx.item() // window_size, idx.item() % window_size
            coordinates['min'].append((y + dy, x + dx))
            thresholded_values['min'].append(1 if val > 0.5 else 0)

    return coordinates, thresholded_values

'''
results = []
for c in range(C):
    # 第一个张量的滑窗和
    sum_tensor = sliding_window_sum(tensor1[c], g)
    # 找到和最大的k个滑窗
    y_indices, x_indices = top_k_windows(sum_tensor, k)
    # 在第二个张量中找到对应的滑窗并获取坐标和阈值处理后的值
    coords, vals = get_values_coordinates_and_threshold(tensor2[c], y_indices, x_indices, n, g)
    results.append((coords, vals))
'''
# 对每个通道进行操作

def frequency_grid_mask(tensor1,k=5,g=64,n=2):
    """
            参数	含义
            k	    筛选响应最大的 k 个无重叠滑窗
            g	    滑窗的大小（window_size）
            n	    每个滑窗内提取最大 n 个点 + 最小 n 个点
            tensor1	用于筛选关键窗口的特征图（比如频域特征、激活图）
            tensor2	用于提取点提示的特征图（和 tensor1 尺寸完全一致）
    """
    C,h,w = tensor1.size()
    results = []
    masks = []
    for c in range(C):
    # 第一个张量的滑窗和
        sum_tensor = sliding_window_sum(tensor1[c], g)
    # 找到和最大的k个滑窗
        y_indices, x_indices = top_k_windows(sum_tensor, k,g)
    # 生成 32×32 掩码
        mask = frequency_mask(y_indices,x_indices,g,g)
        masks.append(mask)

   
    frequency_mask_res = torch.stack(masks)
    return frequency_mask_res

def point_prompt(tensor1,tensor2,k=5,g=32,n=2):
    C,h,w = tensor1.size()
    results = []
    #masks = []
    for c in range(C):
    # 第一个张量的滑窗和
        sum_tensor = sliding_window_sum(tensor1[c], g)
    # 找到和最大的k个滑窗
        y_indices, x_indices = top_k_windows(sum_tensor, k,g)
        #mask = frequency_mask(y_indices,x_indices,g,g)
        #masks.append(mask)
    # 在第二个张量中找到对应的滑窗并获取坐标
        coordinates,vals = get_values_coordinates_and_threshold(tensor2[c], y_indices, x_indices, n, g)
        results.append((coordinates,vals))

    point_final_positions = []
    point_final_values = []
    for c in range(C):
        point_final_positions.append(results[c][0]['max']+results[c][0]['min'])
        point_final_values.append(results[c][1]['max'] + results[c][1]['min'])
#print(len(point_final[0][0]))
    point_final_positions = torch.tensor(point_final_positions)
    point_final_values = torch.tensor(point_final_values)
    #frequency_mask = torch.stack(masks)
    return point_final_positions,point_final_values
# # 输出结果
# for channel, (coords, vals) in enumerate(results):
#     print(f"Channel {channel}:")
#     for key in coords:
#         print(f"{key.capitalize()} value coordinates:")
#         print(" ", coords[key])
#         print(f"{key.capitalize()} thresholded values:")
#         print(" ", vals[key])


def point_prompt_from_attn(attn_map, foreground_mask, k=5, g=32, n=2):
    """
    从频率注意力图中选点（替代point_prompt，不需要外部频率图）
    
    Args:
        attn_map: 频率注意力图 [B, H, W]，来自FGSAttn_Adapter
        foreground_mask: 粗分割前景掩码 [B, H, W]，用于确定点的正负标签
        k: 选择响应最大的k个窗口
        g: 滑窗大小
        n: 每个窗口内提取最大/最小各n个点
    
    Returns:
        point_final_positions: 点坐标 [B, N, 2]
        point_final_values: 点标签 [B, N]
    """
    B, H, W = attn_map.shape
    device = attn_map.device
    
    all_positions = []
    all_values = []
    
    for b in range(B):
        # 在注意力图上滑窗求和，找到高响应区域
        sum_tensor = sliding_window_sum(attn_map[b], g)
        y_indices, x_indices = top_k_windows(sum_tensor, k, g)
        
        # 在前景掩码上提取点坐标和标签
        coordinates, vals = get_values_coordinates_and_threshold(
            foreground_mask[b], y_indices, x_indices, n, g)
        
        # 合并max和min的点
        positions = coordinates['max'] + coordinates['min']
        values = vals['max'] + vals['min']
        
        all_positions.append(positions)
        all_values.append(values)
    
    # 转为张量
    point_final_positions = torch.tensor(all_positions, device=device)
    point_final_values = torch.tensor(all_values, device=device)
    
    return point_final_positions, point_final_values


def frequency_grid_mask_from_attn(attn_map, k=5, g=64):
    """
    从频率注意力图生成频率网格掩码（替代frequency_grid_mask）
    
    Args:
        attn_map: 频率注意力图 [B, H, W]
        k: 选择响应最大的k个窗口
        g: 滑窗大小
    
    Returns:
        frequency_mask_res: 频率网格掩码 [B, 32, 32]
    """
    B, H, W = attn_map.shape
    masks = []
    
    for b in range(B):
        sum_tensor = sliding_window_sum(attn_map[b], g)
        y_indices, x_indices = top_k_windows(sum_tensor, k, g)
        mask = frequency_mask(y_indices, x_indices, g, g)
        masks.append(mask)
    
    frequency_mask_res = torch.stack(masks)
    return frequency_mask_res
