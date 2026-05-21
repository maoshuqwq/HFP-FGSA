import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as F
import time
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# 添加当前目录和父目录到sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TIPCODE_DIR = os.path.join(CURRENT_DIR, '..')
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, TIPCODE_DIR)

from dataset import FullDataset
import tqdm
import warnings

from segment_anything import build_sam, SamPredictor
from segment_anything import sam_model_registry

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

from adapters.sam_lora_image_encoder import LoRA_Sam

SAM_CKPT = os.path.join(TIPCODE_DIR, 'checkpoint', 'sam_vit_b_01ec64.pth')
sam = sam_model_registry["vit_b"](checkpoint=SAM_CKPT)
sam = sam[0]
model = LoRA_Sam(sam, 4).cuda()

model.load_lora_parameters(SAM_CKPT)

model = model.train()


parser = argparse.ArgumentParser("SAM2-UNet")
parser.add_argument("--hiera_path", default='sam2_hiera_base_plus.pt', type=str,
                    help="path to the sam2 pretrained hiera")
parser.add_argument("--train_image_path", default='train/Image/', type=str,
                    help="path to the image that used to train the model")
parser.add_argument("--train_mask_path", default='train/Masks/', type=str,
                    help="path to the mask file for training")
parser.add_argument('--save_path', default='checkpoint', type=str,
                    help="path to store the checkpoint")
parser.add_argument("--epoch", type=int, default=20,
                    help="training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--batch_size", default=6, type=int)
parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument("--log_interval", default=50, type=int)
args = parser.parse_args()

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def adjust_learning_rate(optimizer, epoch, start_lr):
    if epoch % 20 == 0:
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * 0.1
        print(param_group["lr"])

def main(args):
    # 数据集不再需要加载频率图，FGSAttn_Adapter 内部自提取
    dataset = FullDataset(args.train_image_path, args.train_mask_path, 512, mode='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    total_batches = len(dataloader)
    device = torch.device("cuda")
    optim = opt.AdamW([{"params": model.parameters(), "initia_lr": args.lr}], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)
    os.makedirs(args.save_path, exist_ok=True)
    global_start_t = time.perf_counter()
    ema_step_s = None
    for epoch in range(args.epoch):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        epoch_start_t = time.perf_counter()
        for i, batch in enumerate(dataloader):
            step_start_t = time.perf_counter()
            x = batch['image']
            target1 = batch['label1']
            target2 = batch['label2']
            x = x.to(device)
            target1 = target1.to(device)
            target2 = target2.to(device)
            optim.zero_grad()
            # 不再传入 fre 参数，FGSAttn_Adapter 内部自提取频率信息
            pred0, pred1, pred2 = model(x, 1, 512)
            loss0 = structure_loss(pred0, target1)
            loss1 = structure_loss(pred1, target1)
            loss = loss0 + loss1
            loss.backward()
            optim.step()
            step_s = time.perf_counter() - step_start_t
            if ema_step_s is None:
                ema_step_s = step_s
            else:
                ema_step_s = 0.9 * ema_step_s + 0.1 * step_s

            if (i % args.log_interval) == 0 or (i + 1) == total_batches:
                lr = optim.param_groups[0]["lr"]
                done_batches = i + 1
                remaining_batches = total_batches - done_batches
                eta_epoch_s = remaining_batches * (ema_step_s or 0.0)
                remaining_epochs = args.epoch - (epoch + 1)
                eta_total_s = eta_epoch_s + remaining_epochs * total_batches * (ema_step_s or 0.0)

                if torch.cuda.is_available():
                    mem_alloc_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                    mem_reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
                    mem_peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                else:
                    mem_alloc_gb = 0.0
                    mem_reserved_gb = 0.0
                    mem_peak_gb = 0.0

                elapsed_s = time.perf_counter() - global_start_t
                print(
                    "epoch {}/{} batch {}/{} loss {:.6f} (l0 {:.6f} l1 {:.6f}) lr {:.3e} "
                    "step {:.3f}s ema {:.3f}s mem {:.2f}G/{:.2f}G peak {:.2f}G "
                    "eta_epoch {:.1f}m eta_total {:.2f}h elapsed {:.2f}h".format(
                        epoch + 1,
                        args.epoch,
                        done_batches,
                        total_batches,
                        loss.item(),
                        loss0.item(),
                        loss1.item(),
                        lr,
                        step_s,
                        ema_step_s,
                        mem_alloc_gb,
                        mem_reserved_gb,
                        mem_peak_gb,
                        eta_epoch_s / 60.0,
                        eta_total_s / 3600.0,
                        elapsed_s / 3600.0,
                    )
                )

        scheduler.step()
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epoch:
            torch.save(model.state_dict(), os.path.join(args.save_path, 'SAM-512-fps-%d.pth' % (epoch + 1)))
            print('[Saving Snapshot:]', os.path.join(args.save_path, 'SAM-512-fps-%d.pth' % (epoch + 1)))


if __name__ == "__main__":
    main(args)
