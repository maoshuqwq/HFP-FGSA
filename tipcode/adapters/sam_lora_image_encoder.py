from segment_anything import build_sam, SamPredictor
from segment_anything import sam_model_registry
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from segment_anything.modeling import Sam
from safetensors import safe_open
from safetensors.torch import save_file
from adapters.frequency_adapter import FGSAttn_Adapter

from icecream import ic
import torchvision.models as tm

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class deal_mask(nn.Module):
    def __init__(self):
        super().__init__()
        #self.vgg = tm.vgg16_bn(pretrained=True).features[:43]
        self.up = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            LayerNorm2d(256),
            nn.GELU())
    def forward(self,x):
        #x = self.vgg(x)
        x = self.up(x)
        return x

class _LoRA_qkv(nn.Module):

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv


class LoRA_Sam(nn.Module):

    def __init__(self, sam_model: Sam, r: int, lora_layer=None):
        super(LoRA_Sam, self).__init__()
        self.deal_mask = deal_mask()
        # 使用 FGSAttn_Adapter 替代原来的 fre_adapter
        self.fre_adapter = nn.ModuleList([FGSAttn_Adapter() for i in range(12)])
        assert r > 0
        # base_vit_dim = sam_model.image_encoder.patch_embed.proj.out_channels
        # dim = base_vit_dim
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(
                range(len(sam_model.image_encoder.blocks)))  # Only apply lora to the image encoder by default
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # lets freeze first
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False
    
        '''
        # Here, we do the surgery
        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
        '''
        self.reset_parameters()
        self.sam = sam_model
        #self.deal_mask =nn.Conv2d(1,1,1,1,0)
        #self.pretrain ="sam_vit_b_01ec64.pth" 
        #self.load_lora_parameters(self.pretrain)
        self.deal_mask = copy.deepcopy(self.sam.mask_decoder)
        self.deal_prompt = copy.deepcopy(self.sam.prompt_encoder)

    def load_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')
        
        state_dict = torch.load(filename)
        
        sam_dict = self.sam.state_dict()
        
        sam_keys = sam_dict.keys()
        

        # load prompt encoder
        prompt_encoder_keys = [k for k in sam_keys if 'prompt_encoder' in k]
        prompt_encoder_values = [state_dict[k] for k in prompt_encoder_keys]
        prompt_encoder_new_state_dict = {k: v for k, v in zip(prompt_encoder_keys, prompt_encoder_values)}
        sam_dict.update(prompt_encoder_new_state_dict)

        # load mask decoder
        mask_decoder_keys = [k for k in sam_keys if 'mask_decoder' in k and 'mask_decoder.mask_tokens' not in k and 'desam' not in k]
        mask_decoder_values = [state_dict[k] for k in mask_decoder_keys]
        mask_decoder_new_state_dict = {k: v for k, v in zip(mask_decoder_keys, mask_decoder_values)}
        sam_dict.update(mask_decoder_new_state_dict)
      
        self.sam.load_state_dict(sam_dict)
        self.deal_mask = copy.deepcopy(self.sam.mask_decoder)
        self.deal_prompt = copy.deepcopy(self.sam.prompt_encoder)
    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, batched_input, multimask_output, image_size):
        #print(self.deal_prompt.mask_downscaling[0].bias.data)
        #print(self.sam.prompt_encoder.mask_downscaling[0].bias.data)
        return self.sam(self.deal_mask, self.deal_prompt, self.fre_adapter, 
                       batched_input, multimask_output, image_size)['low_res_logits'] ##


