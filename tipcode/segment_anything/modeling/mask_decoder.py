# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
from icecream import ic

from typing import List, Tuple, Type

from .common import LayerNorm2d
from .vmamba import VSSBlock

class senet(nn.Module):
    def __init__(self,c=256,r=16):
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

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.GELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class DeConv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.ConvTranspose2d(c1, c2, 2, 2, 0)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.GELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=7, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s),g=c1)
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=c_)
        #self.add = shortcut and c1 == c2

    def forward(self, x):
        return x+self.cv2(self.cv1(x))# if self.add else self.cv2(self.cv1(x))

class conv_block(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        self.mamba =  nn.Sequential(VSSBlock(hidden_dim=out_c),VSSBlock(hidden_dim=out_c))
        #self.se = senet(c = out_c)
        self.pw1 = Conv(in_c,out_c,1,1)
        #self.dw = CrossConv(c1=out_c,c2=out_c)
        self.pw2 = Conv(out_c,out_c,1,1)
    def forward(self,x):
        x = self.pw1(x)
        x = x.permute(0,2,3,1)
        x = self.mamba(x)
        x = x.permute(0,3,1,2)
        x = self.pw2(x)
        #x = self.se(x)
        return x


class conv_up(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        #self.se = senet(c = out_c)
        self.conv_up = nn.ConvTranspose2d(in_c,out_c,2,2,0)
        self.conv_fu = Conv(out_c,out_c)
    def forward(self,x):
        x = self.conv_up(x)
        x = self.conv_fu(x)
        return x


import torch
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image


def show_feature_channels_colormap(fm, save_prefix="feature", colormap='jet'):
    try:
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "show_feature_channels_colormap 需要 matplotlib。请先执行：pip install matplotlib"
        ) from e

    fm = fm.mean(dim=1)
    fm = fm.cpu()

    num_show = 1
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

class desam(nn.Module):
    def __init__(self,):
        super().__init__()

        self.up1 = conv_up(32,16)
        self.up2 = conv_up(16,1)
        #self.up3 = conv_up(64,64)

        self.fu1 = conv_block(32,32)
        self.fu2 = conv_block(16,16)
        #self.fu3 = conv_block(256,64)
        #self.fu4 = conv_block(128,32)
     
        #self.pre4 = conv_pre(32)
        #self.pre3 = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2),conv_up_pre(64,32)) 
        #self.pre2 = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=4),conv_up_pre(128,32)) 
        #self.pre1 = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=8),conv_up_pre(256,32))   

        #self.pre_final = nn.Conv2d(8,2,1,1,0) 

        
    def forward(self,mask_in):
        mask_in = self.fu1(mask_in)
        mask_in = self.up1(mask_in)
        #show_feature_channels_colormap(mask_in.mean(dim=1))
        mask_in = self.fu2(mask_in)
        mask_out = self.up2(mask_in)
        return mask_out


class conv_up_pre(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        #self.se = senet(c = out_c)
        self.up = Conv(in_c,out_c)
        self.pre = nn.Conv2d(out_c,2,1,1,0)
        #self.conv = CrossConv(in_c,out_c)
    def forward(self,x):
        x = self.up(x)
        x = self.pre(x)
        return x

class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        enable_desam: bool = True,
    ) -> None:
       
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = 2#num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )
        # Whether to run the (FVM) refinement head. Default keeps original behavior.
        self.enable_desam = enable_desam
        self.desam = desam()
        '''
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, 4, iou_head_depth
        )
        '''
    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        masks, iou_pred,new_mask = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        # if multimask_output:
        #     mask_slice = slice(1, None)
        # else:
        #     mask_slice = slice(0, 1)
        # masks = masks[:, mask_slice, :, :]
        # iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred,new_mask

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        #print(src.size())6,256,32,32  +   6,256,32,32   
        if dense_prompt_embeddings.size(0) == 1:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings
        #print(src.max(),src.min())
        #print(dense_prompt_embeddings.max(),dense_prompt_embeddings.min())
        src = src + dense_prompt_embeddings
        
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)  #b,32,128,128
        if self.enable_desam:
            out_new = self.desam(upscaled_embedding)
        else:
            # Keep API stable while skipping heavy compute.
            out_new = None
        #print(upscaled_embedding.size())
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)  # [b, c, token_num]

        b, c, h, w = upscaled_embedding.shape  # [h, token_num, h, w]
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)  # [1, 4, 256, 256], 256 = 4 * 64, the size of image embeddings

        # Generate mask quality predictions
        iou_pred = 0#self.iou_prediction_head(iou_token_out)

        return masks, iou_pred,out_new


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
