# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
from icecream import ic
import copy
from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .frequency_final_point import *
import numpy as np
from torchprofile import profile_macs
import time
import torch

def find_max_k_in_grid(img_tensor, g, k, t = 1):
    #print(img_tensor.size())
    C, H, W = img_tensor.shape  # 图像的通道数、高度和宽度
    grid_h, grid_w = H // g, W // g  # 计算每个小格子的高度和宽度

      # 存储每个格子的最大值及其位置
    final_pos = []
    final_val = []

    # 遍历每个通道
    for c in range(C):
        max_values_positions = []  # 存储最大值位置
        max_values_ = []           # 存储最大值对应的标签
        min_values_positions = []  # 存储最小值位置
        min_values_ = []           # 存储最小值对应的标签
        
        # 遍历每个网格单元
        for i in range(g):
            for j in range(g):
                # 提取当前网格单元：img_tensor[c, 行范围, 列范围]
                grid = img_tensor[c, i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
                
                # 将网格展平为一维向量
                grid_channel = grid.reshape(-1)
                
                # 找到最大的k个值及其索引
                max_vals, max_indices = torch.topk(grid_channel, k)
                # 找到最小的k个值（通过取负后找最大实现）
                min_vals, min_indices = torch.topk(-grid_channel, k)
                
                # 将一维索引转换为二维坐标
                for idx in range(len(max_indices)):
                    # 计算最大值的二维坐标
                    row = max_indices[idx] // grid_w + i*grid_h  # 行坐标
                    col = max_indices[idx] % grid_w + j*grid_w   # 列坐标
                    max_values_positions.append((row.item(), col.item()))
                    
                    # 根据值大小确定标签：大于0.5为正样本(t)，否则为负样本(1-t)
                    if max_vals[idx] > 0.5:
                        max_values_.append(t)
                    else:
                        max_values_.append(abs(1-t))

                    # 计算最小值的二维坐标
                    row = min_indices[idx] // grid_w + i*grid_h
                    col = min_indices[idx] % grid_w + j*grid_w
                    min_values_positions.append((row.item(), col.item()))
                    
                    # 根据值大小确定标签：小于0.5为负样本(1-t)，否则为正样本(t)
                    if -min_vals[idx] < 0.5:
                        min_values_.append(abs(1-t))
                    else:
                        min_values_.append(t)
        
        # 将当前通道的最大/最小值位置和标签合并
        temp_pos = torch.cat([torch.tensor(max_values_positions), torch.tensor(min_values_positions)], dim=0)
        temp_val = torch.cat([torch.tensor(max_values_), torch.tensor(min_values_)], dim=0)
        
        # 添加到最终列表
        final_pos.append(temp_pos)
        final_val.append(temp_val)

    # 将所有通道的结果堆叠成张量
    te = torch.stack(final_pos)
    tw = torch.stack(final_val)
    
    return te, tw


class Sam(nn.Module):
    """
    HFP-SAM (Frequency-Guided Prompt SAM) 主模型类
    
    这是对原生SAM的扩展，引入了频率引导的提示机制，实现两阶段分割：
    1. 粗分割阶段：利用频率信息进行初步分割
    2. 精分割阶段：基于频率引导的点提示进行精细化分割
    
    类常量：
        mask_threshold (float): 掩码二值化阈值，默认0.0
        image_format (str): 输入图像格式，固定为RGB
    """
    # 类常量：掩码阈值，用于将模型输出的logits转换为二值掩码（大于阈值为前景）
    mask_threshold: float = 0.0
    # 类常量：模型输入的图像格式，固定为RGB三通道
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        初始化HFP-SAM模型

        参数:
            image_encoder (ImageEncoderViT): 图像编码器，将图像转换为特征嵌入
            prompt_encoder (PromptEncoder): 提示编码器，将点、框、掩码提示编码为特征
            mask_decoder (MaskDecoder): 掩码解码器，根据图像嵌入和提示生成掩码
            pixel_mean (List[float]): 图像归一化均值，默认使用ImageNet均值
            pixel_std (List[float]): 图像归一化标准差，默认使用ImageNet标准差
        """
        super().__init__()
        
        # 核心组件赋值
        self.image_encoder = image_encoder      # 图像编码器（ViT-H/14）
        self.prompt_encoder = prompt_encoder    # 提示编码器
        self.mask_decoder = mask_decoder        # 掩码解码器（第一阶段）
        
        # HFP-SAM新增：深拷贝掩码解码器用于第二阶段精分割
        # 这样可以让两个阶段的解码器参数独立优化
        self.mask_de128 = copy.deepcopy(self.mask_decoder)
        
        # 注册图像归一化参数（作为缓冲区，不参与训练）
        # 形状转换为 [C, 1, 1]，便于广播到任意大小的图像张量
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        """获取模型所在的设备（CPU/GPU）"""
        # 从已注册的pixel_mean缓冲区获取设备信息
        # 这是一种安全的获取设备方式，确保返回的是模型实际所在的设备
        return self.pixel_mean.device

    def forward(
        self,
        deal_mask: torch.nn.Module,
        deal_prompt: torch.nn.Module,
        fre_adapter: torch.nn.Module,
        batched_input,
        multimask_output,
        image_size
    ) -> Dict[str, torch.Tensor] or List[Dict[str, torch.Tensor]]:
        """
        HFP-SAM主入口函数，根据输入类型自动选择训练或测试模式
        
        参数:
            deal_mask (torch.nn.Module): 第二阶段精分割使用的掩码解码器（深拷贝）
            deal_prompt (torch.nn.Module): 第二阶段精分割使用的提示编码器（深拷贝）
            fre_adapter (torch.nn.Module): 频率适配器模块列表（12个，对应ViT的12层）
            batched_input: 批量输入数据，list类型为测试模式，tensor类型为训练模式
            multimask_output (bool): 是否输出多个掩码
            image_size (int): 输入图像的尺寸（假设为正方形）
        
        返回:
            outputs: 训练模式返回字典，测试模式返回字典列表
        """
        # 根据输入类型判断模式：
        # - list类型：测试模式（原生SAM格式，每个元素是一个字典）
        # - tensor类型：训练模式（HFP-SAM专用格式）
        if isinstance(batched_input, list):
            outputs = self.forward_test(batched_input, multimask_output)
        else:
            outputs = self.forward_train(deal_mask, deal_prompt, fre_adapter, 
                                        batched_input, multimask_output, image_size)
        return outputs

    def forward_train(
        self,
        deal_mask: torch.nn.Module,
        deal_prompt: torch.nn.Module,
        fre_adapter: torch.nn.Module,
        batched_input,
        multimask_output,
        image_size
    ) -> Dict[str, torch.Tensor]:
        """
        HFP-SAM训练阶段的前向传播，实现两阶段频率引导分割：
        
        阶段一（粗分割）：
            1. 使用带频率适配器的图像编码器提取特征（内部做FFT）
            2. 在无提示情况下进行初步分割
        
        阶段二（精分割）：
            1. 根据频率注意力图和粗分割结果自动选择关键点提示
            2. 使用关键点提示进行精细化分割
        
        参数:
            deal_mask (torch.nn.Module): 第二阶段精分割的掩码解码器
            deal_prompt (torch.nn.Module): 第二阶段精分割的提示编码器
            fre_adapter (torch.nn.Module): 频率适配器模块（12层）
            batched_input (torch.Tensor): 批量输入图像张量
            multimask_output (bool): 是否输出多个掩码
            image_size (int): 图像尺寸
        
        返回:
            Dict[str, torch.Tensor]: 包含粗分割掩码、IoU预测和多尺度logits的字典
        """
        # ========== 输入预处理 ==========
        input_images = batched_input

        # ========== 阶段一：粗分割（Coarse Segmentation） ==========
        
        # 1. 使用带频率适配器的图像编码器提取特征
        # FGSAttn_Adapter 在ViT每层后内部做FFT，返回特征和注意力图
        image_embeddings, attention_maps = self.image_encoder(input_images, fre_adapter)
        
        # 2. 编码空提示（无点、框、掩码提示）
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None, boxes=None, masks=None,
        )
        
        # 3. 第一阶段掩码解码（粗分割）
        low_res_masks1, iou_predictions, new_mask = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output
        )

        # 提取前景通道的掩码作为主分割结果
        low_res_masks_b = low_res_masks1[:, 1, :, :].unsqueeze(1)
        
        # 对粗分割结果进行softmax归一化
        mask_prompt_deal = torch.softmax(low_res_masks1, dim=1)
        
        # 获取硬掩码
        mask_prompt_01 = torch.argmax(mask_prompt_deal, axis=1).unsqueeze(1)
        
        # 将低分辨率掩码上采样到原始图像尺寸
        masks_prompt_deal = self.postprocess_masks(
            low_res_masks1,
            input_size=(image_size, image_size),
            original_size=(image_size, image_size)
        )

        # ========== 频率引导点选择（FPS） ==========
        
        # 提取前景通道的掩码用于点选择
        foreground_mask = masks_prompt_deal[:, 1, :, :]
        
        # 取最后一层的频率注意力图（也可用所有层的平均或加权）
        # attention_maps[-1] 形状为 [B, H, W]，H=W=64
        fre_attn = attention_maps[-1]
        
        # 上采样注意力图到与foreground_mask相同尺寸
        fre_attn_resized = F.interpolate(
            fre_attn.unsqueeze(1), 
            size=foreground_mask.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        ).squeeze(1)
        
        # 使用频率注意力图选择关键点
        grid_point, grid_value = point_prompt_from_attn(fre_attn_resized, foreground_mask)
        
        # 将关键点数据移动到GPU
        point_yty = [grid_point.cuda(), grid_value.cuda()]

        # ========== 阶段二：精分割（Fine Segmentation） ==========
        
        # 使用第二阶段的提示编码器编码关键点
        sparse_embeddings, dense_embeddings = deal_prompt(
            points=point_yty, boxes=None, masks=mask_prompt_01.float(),
        )
        
        # 使用第二阶段的掩码解码器进行精分割
        low_res_masks2, iou_predictions, new_mask = deal_mask(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output
        )
        
        # 提取精分割的前景掩码
        low_res_masks_b2 = low_res_masks2[:, 1, :, :].unsqueeze(1)
        
        # 将精分割结果上采样到原始尺寸
        masks = self.postprocess_masks(
            low_res_masks2,
            input_size=(image_size, image_size),
            original_size=(image_size, image_size)
        )

        # ========== 构建输出字典 ==========
        outputs = {
            'masks': masks_prompt_deal,                    # 粗分割结果（用于监督）
            'iou_predictions': iou_predictions,            # IoU预测值
            'low_res_logits': [low_res_masks_b, low_res_masks_b2, new_mask]  # 多尺度logits
        }
        return outputs

    @torch.no_grad()
    def forward_test(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        HFP-SAM测试阶段的前向传播，保持与原生SAM完全兼容
        
        此方法不使用频率适配器和频率引导点选择，完全遵循原生SAM的推理流程：
        1. 图像预处理（归一化+padding）
        2. 图像编码（不使用频率适配器）
        3. 提示编码（点/框/掩码提示）
        4. 掩码解码
        5. 后处理（上采样+二值化）
        
        参数:
            batched_input (List[Dict[str, Any]]): 批量输入数据，每个元素是一个字典，包含：
                - "image": 输入图像张量
                - "original_size": 原始图像尺寸
                - 可选: "point_coords", "point_labels", "boxes", "mask_inputs"
            multimask_output (bool): 是否输出多个掩码（通常为True）
        
        返回:
            List[Dict[str, torch.Tensor]]: 每个元素包含：
                - "masks": 二值化后的分割掩码
                - "iou_predictions": 预测的IoU值
                - "low_res_logits": 低分辨率掩码logits
        """
        # ========== 图像预处理 ==========
        # 对批量中的每张图像进行预处理（归一化 + padding到正方形）
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        
        # ========== 图像编码 ==========
        # 测试阶段不使用频率适配器，与原生SAM行为一致
        image_embeddings = self.image_encoder(input_images)

        # ========== 逐图像推理 ==========
        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            # 提取点提示（如果存在）
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            
            # 编码提示（点、框、掩码）
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            
            # 掩码解码
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),  # 添加batch维度
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            
            # 后处理：上采样到原始尺寸
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],   # 模型输入尺寸
                original_size=image_record["original_size"],    # 原始图像尺寸
            )
            
            # 二值化：大于阈值为前景（1），否则为背景（0）
            masks = masks > self.mask_threshold
            
            # 收集输出
            outputs.append(
                {
                    "masks": masks,              # 二值化掩码
                    "iou_predictions": iou_predictions,  # IoU预测
                    "low_res_logits": low_res_masks,     # 低分辨率logits（用于后续处理）
                }
            )
        
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        掩码后处理：移除padding并上采样到原始图像尺寸
        
        处理流程：
        1. 将掩码上采样到模型输入尺寸（通常为1024x1024）
        2. 裁剪掉padding区域，恢复到实际输入图像尺寸
        3. 上采样到原始图像尺寸
        
        参数:
            masks (torch.Tensor): 掩码解码器输出的低分辨率掩码，形状为 [B, C, H, W]
            input_size (Tuple[int, int]): 输入模型的图像尺寸（经过padding后的尺寸）
            original_size (Tuple[int, int]): 原始图像尺寸（padding前的尺寸）
        
        返回:
            torch.Tensor: 处理后的掩码，形状为 [B, C, original_H, original_W]
        """
        # 步骤1: 将掩码上采样到模型输入尺寸（例如1024x1024）
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",          # 双线性插值，保持平滑
            align_corners=False,      # 不强制对齐角点，避免边界伪影
        )
        
        # 步骤2: 裁剪掉padding区域，恢复到实际输入图像尺寸
        masks = masks[..., :input_size[0], :input_size[1]]
        
        # 步骤3: 上采样到原始图像尺寸
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        图像预处理：归一化像素值并padding到正方形输入
        
        处理流程：
        1. 使用ImageNet均值和标准差进行归一化
        2. 计算padding量，将图像pad到模型要求的正方形尺寸
        
        参数:
            x (torch.Tensor): 输入图像张量，形状为 [C, H, W]
        
        返回:
            torch.Tensor: 预处理后的图像张量，形状为 [C, img_size, img_size]
        """
        # 步骤1: 颜色归一化（使用ImageNet的均值和标准差）
        x = (x - self.pixel_mean) / self.pixel_std

        # 步骤2: Padding到正方形
        h, w = x.shape[-2:]                    # 获取图像高度和宽度
        padh = self.image_encoder.img_size - h  # 垂直方向需要padding的量
        padw = self.image_encoder.img_size - w  # 水平方向需要padding的量
        x = F.pad(x, (0, padw, 0, padh))       # 只在右侧和底部padding
        
        return x

