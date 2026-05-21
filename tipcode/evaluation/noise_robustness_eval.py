"""
Synthetic noise robustness evaluation for SAM vs. SAM+FGA.

Goal (for rebuttal): quantify how segmentation performance degrades when adding
Gaussian / Speckle noise to input images, and show that frequency-guided adapter
(FGA) improves robustness compared to standard SAM.

This script is inference-only (no retraining).

Usage example:
python3 tipcode/noise_robustness_eval.py \
  --ds MAS3K=/Users/yty/Desktop/TIP-HFP-SAM/tipcode/test/Image,/Users/yty/Desktop/TIP-HFP-SAM/tipcode/test/Masks \
  --ds RMAS=/Users/yty/Desktop/TIP-HFP-SAM/rmas_data/test/img,/Users/yty/Desktop/TIP-HFP-SAM/rmas_data/test/labels \
  --ds UFO120=/Users/yty/Desktop/TIP-HFP-SAM/data-ufo/TEST/hr,/Users/yty/Desktop/TIP-HFP-SAM/data-ufo/TEST/masks \
  --ds RUWI=/Users/yty/Desktop/TIP-HFP-SAM/RUWI_DATASET/test_uwi/images,/Users/yty/Desktop/TIP-HFP-SAM/RUWI_DATASET/test_uwi/masks \
  --sam_ckpt /Users/yty/Desktop/TIP-HFP-SAM/tipcode/sam_vit_b_01ec64.pth \
  --fga_ckpt /path/to/your/HFP-SAM_checkpoint.pth \
  --noise gaussian,speckle \
  --sigmas 0,0.05,0.1,0.2 \
  --max_images 300 \
  --out_csv /Users/yty/Desktop/TIP-HFP-SAM/noise_robustness_results.csv
"""

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from frequency_adapter import fre_adapter
from segment_anything.modeling.frequency_final_point import frequency_grid_mask


IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


class NullFreAdapter(nn.Module):
    def forward(self, x: torch.Tensor, fre_mask: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return x


def _parse_list(s: str) -> List[str]:
    return [p.strip() for p in s.split(",") if p.strip()]


def _parse_float_list(s: str) -> List[float]:
    out: List[float] = []
    for p in _parse_list(s):
        out.append(float(p))
    return out


def _list_pairs(image_dir: Path, mask_dir: Path) -> List[Tuple[Path, Path]]:
    imgs: List[Path] = []
    for ext in IMG_EXTS:
        imgs.extend(image_dir.glob(f"*{ext}"))
    imgs = sorted(imgs)
    pairs: List[Tuple[Path, Path]] = []
    for img in imgs:
        m = mask_dir / f"{img.stem}.png"
        if m.exists():
            pairs.append((img, m))
    return pairs


def _read_rgb_uint8(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _read_mask_bool(path: Path) -> np.ndarray:
    m = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    return m > 127


def add_gaussian(img01: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, sigma, size=img01.shape).astype(np.float32)
    out = img01 + noise
    return np.clip(out, 0.0, 1.0)


def add_speckle(img01: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, sigma, size=img01.shape).astype(np.float32)
    out = img01 + img01 * noise
    return np.clip(out, 0.0, 1.0)


def _bbox_xyxy_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / (union + 1e-12))


def _haar_dwt_level1(channel: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = channel.shape
    pad_h = h % 2
    pad_w = w % 2
    if pad_h or pad_w:
        channel = np.pad(channel, ((0, pad_h), (0, pad_w)), mode="edge")
    a = channel[0::2, 0::2]
    b = channel[0::2, 1::2]
    c = channel[1::2, 0::2]
    d = channel[1::2, 1::2]
    ll = (a + b + c + d) / 4.0
    lh = (a - b + c - d) / 4.0
    hl = (a + b - c - d) / 4.0
    hh = (a - b - c + d) / 4.0
    return ll.astype(np.float32), lh.astype(np.float32), hl.astype(np.float32), hh.astype(np.float32)


def frequency_map_from_dhwt(img_rgb_uint8: np.ndarray, rectify: str = "relu") -> np.ndarray:
    x = img_rgb_uint8.astype(np.float32, copy=False)
    mh_channels = []
    for ch in range(3):
        _, lh, hl, hh = _haar_dwt_level1(x[:, :, ch])
        mh_channels.append((lh + hl + hh) / 3.0)
    mh_half = (mh_channels[0] + mh_channels[1] + mh_channels[2]) / 3.0
    rectify = rectify.lower()
    if rectify == "none":
        pass
    elif rectify == "relu":
        mh_half = np.maximum(mh_half, 0.0)
    elif rectify == "abs":
        mh_half = np.abs(mh_half)
    else:
        raise ValueError("rectify must be none/relu/abs")
    return mh_half.astype(np.float32, copy=False)


def resize_float_map(freq: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    h, w = hw
    img_f = Image.fromarray(freq.astype(np.float32, copy=False), mode="F")
    img_r = img_f.resize((w, h), resample=Image.BILINEAR)
    return np.array(img_r, dtype=np.float32)


def build_fre_mask_from_image(img_rgb_uint8: np.ndarray, size: int, device: torch.device) -> torch.Tensor:
    # compute M^h (half-res), then resize to size x size
    mh = frequency_map_from_dhwt(img_rgb_uint8, rectify="relu")
    mh = resize_float_map(mh, (size, size))
    # normalize to [0,1] for stability (ranking unaffected)
    mh = mh - float(mh.min())
    mh = mh / float(mh.max() - mh.min() + 1e-12)
    # NOTE: frequency_grid_mask uses Python slicing, which requires CPU integer indices.
    # Compute on CPU, then move the final mask to the target device.
    fre_cpu = torch.from_numpy(mh).to(dtype=torch.float32).unsqueeze(0)  # [1,H,W] on CPU
    fre_mask = frequency_grid_mask(fre_cpu)  # [1,32,32] on CPU
    return fre_mask.to(device=device, dtype=torch.float32)


def build_models(
    sam_ckpt: str,
    fga_ckpt: Optional[str],
    device: torch.device,
) -> Tuple[torch.nn.Module, torch.nn.ModuleList, Optional[torch.nn.ModuleList]]:
    sam, _ = sam_model_registry["vit_b"](checkpoint=sam_ckpt)
    sam = sam.to(device)
    sam.eval()

    null_adapter = nn.ModuleList([NullFreAdapter() for _ in range(12)]).to(device)

    fga_adapter: Optional[nn.ModuleList] = None
    if fga_ckpt:
        state = torch.load(fga_ckpt, map_location="cpu")
        fga_adapter = nn.ModuleList([fre_adapter() for _ in range(12)])
        # filter fre_adapter weights
        sub = {k.replace("fre_adapter.", "", 1): v for k, v in state.items() if k.startswith("fre_adapter.")}
        missing, unexpected = fga_adapter.load_state_dict(sub, strict=False)
        if missing or unexpected:
            # strict=False is intentional to be robust to different checkpoint layouts.
            # But we still warn to make sure users notice potential mismatch.
            print("[WARN] fre_adapter load_state_dict:", "missing=", len(missing), "unexpected=", len(unexpected))
        fga_adapter = fga_adapter.to(device)
        fga_adapter.eval()

    return sam, null_adapter, fga_adapter


@torch.no_grad()
def predict_with_box(
    sam,
    fre_adapter: nn.ModuleList,
    image_rgb_uint8: np.ndarray,
    box_xyxy: np.ndarray,
    fre_mask: Optional[torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    """
    Returns a boolean mask in original resolution.
    """
    H0, W0 = image_rgb_uint8.shape[:2]
    transform = ResizeLongestSide(sam.image_encoder.img_size)

    # image -> model input (long-side resize)
    input_image = transform.apply_image(image_rgb_uint8)
    input_image_t = torch.as_tensor(input_image, device=device).permute(2, 0, 1).contiguous()[None, ...]
    input_size = tuple(input_image_t.shape[-2:])

    # preprocess (normalize + pad)
    input_image_t = sam.preprocess(input_image_t)

    # fre mask (for FGA). If None, use zeros.
    if fre_mask is None:
        fre_mask = torch.zeros((1, 32, 32), device=device, dtype=torch.float32)

    # image embedding
    image_embeddings = sam.image_encoder(input_image_t, fre_mask, fre_adapter)

    # transform box to input frame
    box_in = transform.apply_boxes(box_xyxy[None, :], (H0, W0))
    box_t = torch.as_tensor(box_in, dtype=torch.float32, device=device)

    # prompt embeddings
    sparse_embeddings, dense_embeddings = sam.prompt_encoder(points=None, boxes=box_t, masks=None)

    # mask prediction
    out = sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    if isinstance(out, (tuple, list)) and len(out) >= 1:
        low_res_masks = out[0]
    else:
        low_res_masks = out

    masks = sam.postprocess_masks(low_res_masks, input_size=input_size, original_size=(H0, W0))
    # binary label: argmax over channels
    pred = masks.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
    return pred.astype(bool)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", action="append", required=True, help="NAME=IMAGE_DIR,MASK_DIR")
    ap.add_argument("--sam_ckpt", type=str, default=str(Path(__file__).parent / "sam_vit_b_01ec64.pth"))
    ap.add_argument("--fga_ckpt", type=str, default="", help="LoRA_Sam checkpoint with fre_adapter weights")
    ap.add_argument("--noise", type=str, default="gaussian,speckle", help="comma-separated: gaussian,speckle")
    ap.add_argument("--sigmas", type=str, default="0,0.05,0.1,0.2", help="comma-separated noise levels in [0,1]")
    ap.add_argument("--max_images", type=int, default=0, help="0 means use all images")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu", "mps"])
    ap.add_argument("--out_csv", type=str, default="noise_robustness_results.csv")
    args = ap.parse_args()

    noise_types = _parse_list(args.noise)
    sigmas = _parse_float_list(args.sigmas)

    # device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    rng_py = random.Random(args.seed)
    rng_np = np.random.default_rng(args.seed)

    # parse datasets
    datasets: List[Tuple[str, Path, Path]] = []
    for item in args.ds:
        name, rest = item.split("=", 1)
        img_dir_s, mask_dir_s = rest.split(",", 1)
        datasets.append((name.strip(), Path(img_dir_s).expanduser(), Path(mask_dir_s).expanduser()))

    sam, null_adapter, fga_adapter = build_models(args.sam_ckpt, args.fga_ckpt or None, device=device)

    rows: List[Dict[str, object]] = []
    for name, img_dir, mask_dir in datasets:
        pairs = _list_pairs(img_dir, mask_dir)
        if not pairs:
            print(f"[WARN] {name}: no image/mask pairs found in {img_dir} / {mask_dir}")
            continue
        rng_py.shuffle(pairs)
        if args.max_images and args.max_images > 0:
            pairs = pairs[: min(len(pairs), int(args.max_images))]
        print(f"{name}: evaluating {len(pairs)} images")

        for noise_t in noise_types:
            for sigma in sigmas:
                for variant in ("SAM", "SAM+FGA"):
                    if variant == "SAM+FGA" and fga_adapter is None:
                        continue

                    ious: List[float] = []
                    for img_p, gt_p in pairs:
                        img = _read_rgb_uint8(img_p)
                        gt = _read_mask_bool(gt_p)
                        box = _bbox_xyxy_from_mask(gt)
                        if box is None:
                            continue

                        img01 = (img.astype(np.float32) / 255.0)
                        if sigma > 0:
                            if noise_t == "gaussian":
                                img01 = add_gaussian(img01, sigma=sigma, rng=rng_np)
                            elif noise_t == "speckle":
                                img01 = add_speckle(img01, sigma=sigma, rng=rng_np)
                            else:
                                raise ValueError(f"unknown noise type: {noise_t}")
                        img_noisy = (img01 * 255.0 + 0.5).astype(np.uint8)

                        fre_mask = None
                        adapter = null_adapter
                        if variant == "SAM+FGA":
                            adapter = fga_adapter  # type: ignore[assignment]
                            fre_mask = build_fre_mask_from_image(img_noisy, size=512, device=device)

                        pred = predict_with_box(
                            sam=sam,
                            fre_adapter=adapter,
                            image_rgb_uint8=img_noisy,
                            box_xyxy=box,
                            fre_mask=fre_mask,
                            device=device,
                        )
                        ious.append(iou(pred, gt))

                    miou = float(np.mean(ious)) if ious else float("nan")
                    rows.append(
                        {
                            "dataset": name,
                            "variant": variant,
                            "noise": noise_t,
                            "sigma": sigma,
                            "n_images": len(ious),
                            "mIoU": miou,
                        }
                    )
                    print(f"[{name}] {variant} {noise_t} sigma={sigma}: mIoU={miou:.4f} (n={len(ious)})")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["dataset", "variant", "noise", "sigma", "n_images", "mIoU"])
        wr.writeheader()
        wr.writerows(rows)
    print("Wrote:", str(out_csv))


if __name__ == "__main__":
    main()


