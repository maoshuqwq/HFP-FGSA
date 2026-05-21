"""
用 GT mask 对“频域选窗”做定量分析（无需跑模型），用于回复“不同分辨率的超参敏感性分析”。

核心思想：
- 对每张图，在给定分辨率(H×W)和窗口大小(win×win)下，根据频率图 M^h 选 top-k 窗口
- 用 GT mask 判断每个窗口是否同时包含正/负样本（mixed），以及是否覆盖目标边界（boundary hit）
- 额外给出随机选窗 baseline（同样的窗口划分、同样 top-k 数量），对比 ours vs random

数据路径（按你项目结构）：
- Image:  tipcode/train/Image/*.jpg
- Masks:  tipcode/train/Masks/*.png
- (可选) Frequency_2: tipcode/train/Frequency_2/*.jpg

示例：
python3 tipcode/frequency_window_gt_analysis.py \
  --image_dir /Users/yty/Desktop/TIP-HFP-SAM/tipcode/train/Image \
  --mask_dir  /Users/yty/Desktop/TIP-HFP-SAM/tipcode/train/Masks \
  --freq_dir  /Users/yty/Desktop/TIP-HFP-SAM/tipcode/train/Frequency_2 \
  --sizes 256,512,1024 \
  --windows 64,32,16 \
  --topk 10 \
  --freq_source auto \
  --max_images 200 \
  --out_dir /Users/yty/Desktop/TIP-HFP-SAM/freqwin_gt_analysis
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Setting:
    size: int
    window: int


def _get_resampling():
    if hasattr(Image, "Resampling"):
        return Image.Resampling
    return Image  # type: ignore[return-value]


def _parse_int_list(s: str) -> List[int]:
    vals: List[int] = []
    for raw in (p.strip() for p in s.split(",")):
        if not raw:
            continue
        v = int(raw)
        if v <= 0:
            raise ValueError(f"非法正整数: {raw!r}")
        vals.append(v)
    if not vals:
        raise ValueError("列表为空")
    return vals


def _list_image_mask_pairs(image_dir: Path, mask_dir: Path) -> List[Tuple[Path, Path]]:
    # 优先按 jpg 扫描，兼容大小写
    imgs = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.JPG")) + list(image_dir.glob("*.png")))
    pairs: List[Tuple[Path, Path]] = []
    for img in imgs:
        stem = img.stem
        m = mask_dir / f"{stem}.png"
        if m.exists():
            pairs.append((img, m))
    return pairs


def _infer_freq_path(img_path: Path, freq_dir: Optional[Path]) -> Optional[Path]:
    """
    按 stem 匹配 Frequency_2 下的同名 jpg。
    """
    if freq_dir is None:
        return None
    p = freq_dir / f"{img_path.stem}.jpg"
    if p.exists():
        return p
    p2 = freq_dir / f"{img_path.stem}.png"
    if p2.exists():
        return p2
    return None


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


def frequency_map_from_dhwt(img_rgb: np.ndarray, rectify: str = "relu") -> np.ndarray:
    x = img_rgb.astype(np.float32, copy=False)
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
        raise ValueError("rectify 只能是 none/relu/abs")
    return mh_half.astype(np.float32, copy=False)


def resize_float_map(freq: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    h, w = hw
    resampling = _get_resampling()
    img_f = Image.fromarray(freq.astype(np.float32, copy=False), mode="F")
    img_r = img_f.resize((w, h), resample=resampling.BILINEAR)
    return np.array(img_r, dtype=np.float32)


def window_scores(freq_map: np.ndarray, window: int) -> np.ndarray:
    h, w = freq_map.shape
    n_rows = h // window
    n_cols = w // window
    if n_rows <= 0 or n_cols <= 0:
        raise ValueError(f"window={window} 过大，无法在 {h}x{w} 上划分窗口")
    h2 = n_rows * window
    w2 = n_cols * window
    crop = freq_map[:h2, :w2]
    return crop.reshape(n_rows, window, n_cols, window).mean(axis=(1, 3)).astype(np.float32)


def topk_linear_indices(scores: np.ndarray, k: int) -> np.ndarray:
    flat = scores.reshape(-1)
    k = max(0, min(int(k), flat.size))
    if k == 0:
        return np.array([], dtype=np.int64)
    # argpartition + sort（比全排序快）
    idx = np.argpartition(flat, -k)[-k:]
    idx = idx[np.argsort(flat[idx])[::-1]]
    return idx.astype(np.int64, copy=False)


def mask_boundary(mask_bool: np.ndarray) -> np.ndarray:
    """
    用 3x3 erosion 得到边界：boundary = mask & (~erode(mask))。
    不依赖 scipy/cv2。
    """
    m = mask_bool.astype(bool, copy=False)
    h, w = m.shape
    padded = np.pad(m, ((1, 1), (1, 1)), mode="constant", constant_values=False)
    eroded = m.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            eroded &= padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
    return m & (~eroded)


def window_mask_stats(mask_bool: np.ndarray, boundary_bool: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回每个窗口的：
    - pos_frac: (n_rows,n_cols) 正样本占比
    - boundary_hit: (n_rows,n_cols) 是否包含边界像素
    """
    h, w = mask_bool.shape
    n_rows = h // window
    n_cols = w // window
    h2 = n_rows * window
    w2 = n_cols * window
    m = mask_bool[:h2, :w2]
    b = boundary_bool[:h2, :w2]
    area = float(window * window)

    pos_counts = m.reshape(n_rows, window, n_cols, window).sum(axis=(1, 3)).astype(np.float32)
    b_counts = b.reshape(n_rows, window, n_cols, window).sum(axis=(1, 3)).astype(np.int32)
    pos_frac = pos_counts / area
    boundary_hit = b_counts > 0
    return pos_frac, boundary_hit


def summarize_selected(
    pos_frac_flat: np.ndarray,
    boundary_hit_flat: np.ndarray,
    idx: np.ndarray,
    eps: float,
) -> Dict[str, float]:
    if idx.size == 0:
        return {
            "mixed_frac": float("nan"),
            "pure_pos_frac": float("nan"),
            "pure_neg_frac": float("nan"),
            "boundary_hit_frac": float("nan"),
            "pos_frac_mean": float("nan"),
        }
    sel = pos_frac_flat[idx]
    mixed = (sel > eps) & (sel < (1.0 - eps))
    pure_pos = sel >= (1.0 - eps)
    pure_neg = sel <= eps
    b_hit = boundary_hit_flat[idx]
    k = float(idx.size)
    return {
        "mixed_frac": float(mixed.mean()),
        "pure_pos_frac": float(pure_pos.mean()),
        "pure_neg_frac": float(pure_neg.mean()),
        "boundary_hit_frac": float(b_hit.mean()),
        "pos_frac_mean": float(sel.mean()),
    }


def mean_std(x: List[float]) -> Tuple[float, float]:
    arr = np.asarray(x, dtype=np.float32)
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def plot_heatmap(
    mat: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    out_path: Path,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    fig, ax = plt.subplots(figsize=(1.2 * len(col_labels) + 2.0, 0.7 * len(row_labels) + 2.0))
    im = ax.imshow(mat, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="GT-based frequency window analysis (no model run)")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--freq_dir", type=str, default="", help="Frequency_2 目录（可选）")
    parser.add_argument("--sizes", type=str, default="256,512,1024")
    parser.add_argument("--windows", type=str, default="64,32,16")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--eps", type=float, default=0.05, help="pure/mixed 判定阈值：pos_frac<=eps 视为纯负；>=1-eps 视为纯正")
    parser.add_argument(
        "--freq_source",
        type=str,
        default="auto",
        choices=["auto", "dataset", "dhwt"],
        help="auto: 优先用 Frequency_2，否则 dhwt；dataset: 强制用 Frequency_2；dhwt: 从图像算",
    )
    parser.add_argument("--rectify", type=str, default="relu", choices=["none", "relu", "abs"])
    parser.add_argument("--random_trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_images", type=int, default=0, help="0 表示全量；否则只取前 N 张做快速统计")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--ref_size", type=int, default=512, help="用于自适应窗口的参考分辨率（默认512）")
    parser.add_argument("--ref_window", type=int, default=32, help="用于自适应窗口的参考窗口大小（默认32）")

    args = parser.parse_args()
    image_dir = Path(args.image_dir).expanduser().resolve()
    mask_dir = Path(args.mask_dir).expanduser().resolve()
    freq_dir = Path(args.freq_dir).expanduser().resolve() if args.freq_dir else None
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes = _parse_int_list(args.sizes)
    windows = _parse_int_list(args.windows)
    topk = int(args.topk)
    eps = float(args.eps)
    trials = int(args.random_trials)
    rng = np.random.default_rng(int(args.seed))

    pairs = _list_image_mask_pairs(image_dir, mask_dir)
    if int(args.max_images) > 0:
        pairs = pairs[: int(args.max_images)]
    n_images = len(pairs)
    if n_images == 0:
        raise RuntimeError("未找到可匹配的 image/mask 对，请检查目录与文件名。")

    settings = [Setting(size=s, window=w) for s in sizes for w in windows]
    # per setting: collect per-image metrics
    per_setting: Dict[Tuple[int, int], Dict[str, List[float]]] = {}
    for st in settings:
        per_setting[(st.size, st.window)] = {
            "mixed_frac": [],
            "pure_pos_frac": [],
            "pure_neg_frac": [],
            "boundary_hit_frac": [],
            "pos_frac_mean": [],
            "rand_mixed_frac": [],
            "rand_boundary_hit_frac": [],
        }

    resampling = _get_resampling()

    for img_path, mask_path in pairs:
        # 读 GT mask
        m0 = Image.open(str(mask_path)).convert("L")
        mask0 = np.array(m0, dtype=np.uint8) > 0

        # 频率图（dataset）只需要读一次，然后 resize
        freq_pil = None
        freq_path = _infer_freq_path(img_path, freq_dir)
        if args.freq_source in {"auto", "dataset"} and freq_path is not None and freq_path.exists():
            freq_pil = Image.open(str(freq_path)).convert("L")
        if args.freq_source == "dataset" and freq_pil is None:
            # 强制 dataset，但缺失
            continue

        # 若需要 dhwt，则读原图一次
        img_pil = None
        if args.freq_source == "dhwt" or (args.freq_source == "auto" and freq_pil is None):
            img_pil = Image.open(str(img_path)).convert("RGB")

        for st in settings:
            size = st.size
            win = st.window
            hw = (size, size)

            # resize mask（nearest）
            m_res = m0.resize((size, size), resample=resampling.NEAREST)
            mask = np.array(m_res, dtype=np.uint8) > 0
            boundary = mask_boundary(mask)

            # frequency map at this resolution
            if args.freq_source == "dhwt" or (args.freq_source == "auto" and freq_pil is None):
                assert img_pil is not None
                img_res = img_pil.resize((size, size), resample=resampling.BICUBIC)
                rgb = np.array(img_res, dtype=np.uint8)
                mh_half = frequency_map_from_dhwt(rgb, rectify=args.rectify)
                mh = resize_float_map(mh_half, hw)
            else:
                assert freq_pil is not None
                f_res = freq_pil.resize((size, size), resample=resampling.BILINEAR)
                mh = np.array(f_res, dtype=np.float32)

            # scores + select
            scores = window_scores(mh, window=win)
            idx_sel = topk_linear_indices(scores, k=topk)

            # mask stats per window (precompute)
            pos_frac, b_hit = window_mask_stats(mask, boundary, window=win)
            pos_flat = pos_frac.reshape(-1)
            b_flat = b_hit.reshape(-1)

            ours = summarize_selected(pos_flat, b_flat, idx_sel, eps=eps)
            ps = per_setting[(size, win)]
            ps["mixed_frac"].append(ours["mixed_frac"])
            ps["pure_pos_frac"].append(ours["pure_pos_frac"])
            ps["pure_neg_frac"].append(ours["pure_neg_frac"])
            ps["boundary_hit_frac"].append(ours["boundary_hit_frac"])
            ps["pos_frac_mean"].append(ours["pos_frac_mean"])

            # random baseline
            if trials > 0 and idx_sel.size > 0:
                flat_n = scores.size
                k_eff = idx_sel.size
                rm = []
                rb = []
                for _ in range(trials):
                    ridx = rng.choice(flat_n, size=k_eff, replace=False)
                    rr = summarize_selected(pos_flat, b_flat, ridx, eps=eps)
                    rm.append(rr["mixed_frac"])
                    rb.append(rr["boundary_hit_frac"])
                ps["rand_mixed_frac"].append(float(np.mean(rm)))
                ps["rand_boundary_hit_frac"].append(float(np.mean(rb)))
            else:
                ps["rand_mixed_frac"].append(float("nan"))
                ps["rand_boundary_hit_frac"].append(float("nan"))

    # 汇总输出 CSV
    csv_path = out_dir / "freqwin_gt_summary.csv"
    lines = [
        "size,window,topk,freq_source,n_images,eps,"
        "mixed_mean,mixed_std,rand_mixed_mean,rand_mixed_std,delta_mixed,"
        "boundary_mean,boundary_std,rand_boundary_mean,rand_boundary_std,delta_boundary,"
        "pure_pos_mean,pure_neg_mean,pos_frac_mean"
    ]
    for s in sizes:
        for w in windows:
            ps = per_setting[(s, w)]
            mixed_m, mixed_s = mean_std(ps["mixed_frac"])
            rand_m, rand_s = mean_std(ps["rand_mixed_frac"])
            b_m, b_s = mean_std(ps["boundary_hit_frac"])
            rb_m, rb_s = mean_std(ps["rand_boundary_hit_frac"])
            pp_m, _ = mean_std(ps["pure_pos_frac"])
            pn_m, _ = mean_std(ps["pure_neg_frac"])
            pf_m, _ = mean_std(ps["pos_frac_mean"])
            lines.append(
                f"{s},{w},{topk},{args.freq_source},{n_images},{eps:.3f},"
                f"{mixed_m:.4f},{mixed_s:.4f},{rand_m:.4f},{rand_s:.4f},{(mixed_m-rand_m):.4f},"
                f"{b_m:.4f},{b_s:.4f},{rb_m:.4f},{rb_s:.4f},{(b_m-rb_m):.4f},"
                f"{pp_m:.4f},{pn_m:.4f},{pf_m:.4f}"
            )
    csv_path.write_text("\n".join(lines), encoding="utf-8")

    # 画热力图（ours / delta vs random）
    row_labels = [f"{s}x{s}" for s in sizes]
    col_labels = [f"win={w}" for w in windows]

    mixed_mat = np.zeros((len(sizes), len(windows)), dtype=np.float32)
    delta_mixed_mat = np.zeros_like(mixed_mat)
    boundary_mat = np.zeros_like(mixed_mat)
    delta_boundary_mat = np.zeros_like(mixed_mat)

    for i, s in enumerate(sizes):
        for j, w in enumerate(windows):
            ps = per_setting[(s, w)]
            mixed_m, _ = mean_std(ps["mixed_frac"])
            rand_m, _ = mean_std(ps["rand_mixed_frac"])
            b_m, _ = mean_std(ps["boundary_hit_frac"])
            rb_m, _ = mean_std(ps["rand_boundary_hit_frac"])
            mixed_mat[i, j] = mixed_m
            delta_mixed_mat[i, j] = mixed_m - rand_m
            boundary_mat[i, j] = b_m
            delta_boundary_mat[i, j] = b_m - rb_m

    # 额外：用“窗口内前景占比”做分辨率自适应分析（用于展示 256->16, 512->32, 1024->64 这种趋势）
    pos_mat = np.zeros((len(sizes), len(windows)), dtype=np.float32)
    for i, s in enumerate(sizes):
        for j, w in enumerate(windows):
            ps = per_setting[(s, w)]
            pf_m, _ = mean_std(ps["pos_frac_mean"])
            pos_mat[i, j] = pf_m

    # 参考混合度：默认取 (ref_size, ref_window) 的 pos_frac_mean（若不存在则取 512 行中位列兜底）
    ref_size = int(args.ref_size)
    ref_window = int(args.ref_window)
    if (ref_size in sizes) and (ref_window in windows):
        i0 = sizes.index(ref_size)
        j0 = windows.index(ref_window)
        ref_pos = float(pos_mat[i0, j0])
    else:
        # 兜底：取 pos_mat 的中位数
        ref_pos = float(np.nanmedian(pos_mat))

    pos_dev_mat = np.abs(pos_mat - ref_pos)

    plot_heatmap(
        mixed_mat,
        row_labels=row_labels,
        col_labels=col_labels,
        title="Mixed window fraction",
        out_path=out_dir / "heatmap_mixed_ours.png",
        vmin=0.0,
        vmax=1.0,
    )
    plot_heatmap(
        delta_mixed_mat,
        row_labels=row_labels,
        col_labels=col_labels,
        title="Delta vs Random: mixed window fraction",
        out_path=out_dir / "heatmap_mixed_delta.png",
        vmin=float(np.nanmin(delta_mixed_mat)),
        vmax=float(np.nanmax(delta_mixed_mat)),
    )
    plot_heatmap(
        boundary_mat,
        row_labels=row_labels,
        col_labels=col_labels,
        title="Boundary-hit fraction",
        out_path=out_dir / "heatmap_boundary_ours.png",
        vmin=0.0,
        vmax=1.0,
    )
    plot_heatmap(
        delta_boundary_mat,
        row_labels=row_labels,
        col_labels=col_labels,
        title="Delta vs Random: boundary-hit fraction",
        out_path=out_dir / "heatmap_boundary_delta.png",
        vmin=float(np.nanmin(delta_boundary_mat)),
        vmax=float(np.nanmax(delta_boundary_mat)),
    )

    plot_heatmap(
        pos_mat,
        row_labels=row_labels,
        col_labels=col_labels,
        title="Mean foreground ratio in selected top-k windows",
        out_path=out_dir / "heatmap_posfrac_ours.png",
        vmin=0.0,
        vmax=1.0,
    )
    plot_heatmap(
        pos_dev_mat,
        row_labels=row_labels,
        col_labels=col_labels,
        title=f"Abs dev to reference ratio={ref_pos:.2f} (@{args.ref_size}, w={args.ref_window})",
        out_path=out_dir / "heatmap_posfrac_dev.png",
        vmin=float(np.nanmin(pos_dev_mat)),
        vmax=float(np.nanmax(pos_dev_mat)),
    )

    # 折线图：pos_frac_mean vs window（每个分辨率一条曲线），并标出最接近 ref 的 window
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.array(windows, dtype=np.int32)
    for i, s in enumerate(sizes):
        y = pos_mat[i]
        ax.plot(x, y, marker="o", linewidth=2, label=f"{s}x{s}")
        j_best = int(np.nanargmin(pos_dev_mat[i]))
        ax.scatter([x[j_best]], [y[j_best]], s=90, marker="*", zorder=5)
        ax.text(x[j_best], y[j_best] + 0.015, f"w={windows[j_best]}", ha="center", fontsize=9)
    ax.axhline(ref_pos, color="gray", linestyle="--", linewidth=1.5, label=f"ref={ref_pos:.2f}")
    ax.set_xlabel("window size (pixels)")
    ax.set_ylabel("mean foreground ratio")
    ax.set_title("Resolution-aware window size selection via matching reference ratio")
    ax.set_xticks(x)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(str(out_dir / "plot_posfrac_adaptive.png"), dpi=200)
    plt.close(fig)

    print("Saved:")
    print(str(csv_path))
    print(str(out_dir / "heatmap_mixed_ours.png"))
    print(str(out_dir / "heatmap_mixed_delta.png"))
    print(str(out_dir / "heatmap_boundary_ours.png"))
    print(str(out_dir / "heatmap_boundary_delta.png"))
    print(str(out_dir / "heatmap_posfrac_ours.png"))
    print(str(out_dir / "heatmap_posfrac_dev.png"))
    print(str(out_dir / "plot_posfrac_adaptive.png"))


if __name__ == "__main__":
    main()


