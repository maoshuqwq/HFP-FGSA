"""
多分辨率“频域信息 grid”可视化（覆盖到原图上），与论文逻辑对齐。

论文 (main.tex) 的关键流程：
- Eq.(mh): 通过 Haar 小波(DHWT)得到 3 个高频子带，并平均得到频率图 M^h
- Eq.(topw): 在 M^h 上做滑窗统计，选 top-k 响应窗口作为 prior regions P

本脚本做的事情：
- 读取原图 I（RGB）
- 选择频率图来源：
  - auto/dataset：优先使用与 Image 同名的 `Frequency_2/*.jpg`（更贴近训练时用的频率图）
  - dhwt：从当前分辨率的 I 计算一层 Haar 频率图（与 Eq.(mh) 一致）
- 对每个目标分辨率，直接 resize 到指定 H×W（不保持比例）
- 将频率图 resize 到同样的 H×W
- 将频率图划分为 grid×grid 的窗口，计算每个窗口的平均响应并取 top-k
- 把 top-k 窗口矩形框覆盖画到原图上（就像你给的示例图）

示例：
python3 tipcode/visual_frequency_grid.py \
  --image /Users/yty/Desktop/TIP-HFP-SAM/tipcode/train/Image/MAS_Reptile_Turtle_Com_720.jpg \
  --sizes 256,512,1024 \
  --grids 4,8,16,32 \
  --topk 10 \
  --freq_source auto \
  --out_dir /Users/yty/Desktop/TIP-HFP-SAM/frequency_grid_visual
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 为了在无显示环境中也能保存图像
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


@dataclass(frozen=True)
class SizeSpec:
    h: int
    w: int
    label: str


def _normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def _get_resampling():
    # Pillow 10+: Image.Resampling；旧版本兼容 fallback
    if hasattr(Image, "Resampling"):
        return Image.Resampling
    return Image  # type: ignore[return-value]


def _load_times_new_roman(font_size: int) -> ImageFont.ImageFont:
    """
    尽量加载 Times New Roman；若系统缺失则回退到默认字体。
    """
    candidates = [
        # macOS 常见位置
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
        "/Library/Fonts/Times New Roman.ttf",
        "/Library/Fonts/Times New Roman Bold.ttf",
        # Windows 常见位置（以防用户在其它系统跑）
        "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/timesbd.ttf",
    ]
    for p in candidates:
        try:
            if Path(p).exists():
                return ImageFont.truetype(p, font_size)
        except Exception:
            pass

    # 尝试按字体名加载（某些环境可用）
    for name in ["Times New Roman", "Times", "Times-Roman"]:
        try:
            return ImageFont.truetype(name, font_size)
        except Exception:
            continue

    return ImageFont.load_default()


def _resize_pil(img: Image.Image, hw: Tuple[int, int], resample) -> Image.Image:
    h, w = hw
    return img.resize((w, h), resample=resample)


def _parse_sizes(s: str, orig_hw: Tuple[int, int]) -> List[SizeSpec]:
    """
    支持：
    - "orig"
    - "512"（表示 512x512）
    - "512x384"（表示 HxW）
    """
    specs: List[SizeSpec] = []
    for raw in (p.strip() for p in s.split(",")):
        if not raw:
            continue
        lower = raw.lower()
        if lower in {"orig", "original", "raw"}:
            h, w = orig_hw
            specs.append(SizeSpec(h=h, w=w, label=f"orig_{h}x{w}"))
            continue
        if "x" in lower:
            a, b = lower.split("x", 1)
            h = int(a)
            w = int(b)
            if h <= 0 or w <= 0:
                raise ValueError(f"非法尺寸: {raw!r}")
            specs.append(SizeSpec(h=h, w=w, label=f"{h}x{w}"))
            continue
        v = int(lower)
        if v <= 0:
            raise ValueError(f"非法尺寸: {raw!r}")
        specs.append(SizeSpec(h=v, w=v, label=f"{v}x{v}"))

    if not specs:
        raise ValueError("sizes 为空：请至少给一个，比如 '512,1024' 或 'orig,512'")
    return specs


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


def _infer_frequency2_path(image_path: Path) -> Optional[Path]:
    """
    从 Image 路径推断同名 Frequency_2 路径：
    .../train/Image/xxx.jpg  -> .../train/Frequency_2/xxx.jpg
    """
    name = image_path.name
    parent = image_path.parent
    if parent.name.lower() == "image":
        candidate = parent.parent / "Frequency_2" / name
        if candidate.exists():
            return candidate
    # 兜底：同目录下 Frequency_2
    candidate2 = parent / "Frequency_2" / name
    if candidate2.exists():
        return candidate2
    return None


def _haar_dwt_level1(channel: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    一层 2D Haar 小波分解（不依赖第三方库）。
    输入 channel: (H,W) float32
    输出 (LL, LH, HL, HH): 均为 (ceil(H/2), ceil(W/2))

    注：命名 LH/HL 的方向在不同实现中可能互换，但这里最终会对三路高频取平均，因此方向不影响结果。
    """
    h, w = channel.shape
    # padding 到偶数，便于 2x2 分块
    pad_h = (h % 2)
    pad_w = (w % 2)
    if pad_h or pad_w:
        channel = np.pad(channel, ((0, pad_h), (0, pad_w)), mode="edge")
        h, w = channel.shape

    a = channel[0::2, 0::2]
    b = channel[0::2, 1::2]
    c = channel[1::2, 0::2]
    d = channel[1::2, 1::2]

    ll = (a + b + c + d) / 4.0
    lh = (a - b + c - d) / 4.0
    hl = (a + b - c - d) / 4.0
    hh = (a - b - c + d) / 4.0
    return ll, lh, hl, hh


def frequency_map_from_dhwt(
    img_rgb: np.ndarray,  # (H,W,3) uint8/float
    rectify: str = "relu",
) -> np.ndarray:
    """
    与论文 Eq.(mh) 对齐：
    - 对 RGB 每个通道做一层 DHWT 得到 (lh,hl,hh)
    - 取三路高频平均得到 M^h
    - 再对 3 个通道的 M^h 做平均

    返回：
    - mh_half: (H/2, W/2) float32（严格来说是 ceil(H/2), ceil(W/2)）
    """
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError(f"img_rgb 形状应为 (H,W,3)，但得到 {img_rgb.shape}")

    x = img_rgb.astype(np.float32, copy=False)
    mh_channels = []
    for ch in range(3):
        _, lh, hl, hh = _haar_dwt_level1(x[:, :, ch])
        mh = (lh + hl + hh) / 3.0
        mh_channels.append(mh)
    mh_half = (mh_channels[0] + mh_channels[1] + mh_channels[2]) / 3.0

    rectify = rectify.lower()
    if rectify == "none":
        pass
    elif rectify == "relu":
        mh_half = np.maximum(mh_half, 0.0)
    elif rectify == "abs":
        mh_half = np.abs(mh_half)
    else:
        raise ValueError(f"不支持的 rectify: {rectify!r}（可选：none/relu/abs）")

    return mh_half.astype(np.float32, copy=False)


def resize_frequency_map_to_hw(freq: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    """将频率图 resize 到 (H,W)，支持 float32（使用 Pillow F 模式）。"""
    h, w = hw
    resampling = _get_resampling()
    img_f = Image.fromarray(freq.astype(np.float32, copy=False), mode="F")
    img_r = img_f.resize((w, h), resample=resampling.BILINEAR)
    return np.array(img_r, dtype=np.float32)


def grid_scores(
    freq_map: np.ndarray,
    grid: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将 freq_map(H,W) 划分为 grid×grid，并返回：
    - scores: (grid,grid) 每格平均值（对应论文的滑窗平均响应）
    - ys/xs: 边界坐标
    """
    if grid <= 0:
        raise ValueError("grid 必须 > 0")
    h, w = freq_map.shape
    ys = np.linspace(0, h, grid + 1, dtype=np.int32)
    xs = np.linspace(0, w, grid + 1, dtype=np.int32)
    scores = np.zeros((grid, grid), dtype=np.float32)
    for i in range(grid):
        y0, y1 = int(ys[i]), int(ys[i + 1])
        for j in range(grid):
            x0, x1 = int(xs[j]), int(xs[j + 1])
            cell = freq_map[y0:y1, x0:x1]
            scores[i, j] = float(cell.mean()) if cell.size else 0.0
    return scores, ys, xs


def window_scores(
    freq_map: np.ndarray,
    window: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    固定窗口像素大小（window×window），stride=window（非重叠）地做平均响应。
    返回：
    - scores: (H//window, W//window)
    - ys/xs: 边界坐标（0, window, 2*window, ...）

    注：如果 H/W 不能整除 window，会裁剪掉边缘余量（与原始 unfold/stride=window 的实现一致）。
    """
    if window <= 0:
        raise ValueError("window 必须 > 0")
    h, w = freq_map.shape
    n_rows = h // window
    n_cols = w // window
    if n_rows <= 0 or n_cols <= 0:
        raise ValueError(f"window={window} 过大，无法在 {h}x{w} 上划分窗口")

    h2 = n_rows * window
    w2 = n_cols * window
    crop = freq_map[:h2, :w2]
    # (n_rows, window, n_cols, window) -> mean over window dims
    scores = crop.reshape(n_rows, window, n_cols, window).mean(axis=(1, 3)).astype(np.float32)
    ys = (np.arange(0, h2 + 1, window)).astype(np.int32)
    xs = (np.arange(0, w2 + 1, window)).astype(np.int32)
    return scores, ys, xs


def topk_cells(scores: np.ndarray, k: int) -> List[Tuple[int, int]]:
    """返回 top-k 格子坐标 (row, col)。支持任意矩阵形状。"""
    r, c = scores.shape
    k = max(0, min(int(k), r * c))
    if k == 0:
        return []
    flat = scores.reshape(-1)
    idx = np.argsort(flat)[::-1][:k]
    return [(int(i // c), int(i % c)) for i in idx]


def plot_overlay_only(
    rgb: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    topk: List[Tuple[int, int]],
    title: str,
    save_path: Path,
    rect_color: str = "red",
    rect_lw: float = 2.2,
    show: bool = False,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(rgb)
    ax.set_title(title)
    ax.axis("off")

    # 去重画边线，避免相邻/重叠边变粗
    h, w = rgb.shape[:2]

    def _clamp_x(x: int) -> int:
        return int(max(0, min(w - 1, x)))

    def _clamp_y(y: int) -> int:
        return int(max(0, min(h - 1, y)))

    segments = set()
    for (r, c) in topk:
        y0, y1 = int(ys[r]), int(ys[r + 1])
        x0, x1 = int(xs[c]), int(xs[c + 1])
        x0c, x1c = _clamp_x(x0), _clamp_x(x1)
        y0c, y1c = _clamp_y(y0), _clamp_y(y1)
        segs = [
            ((x0c, y0c), (x1c, y0c)),
            ((x0c, y1c), (x1c, y1c)),
            ((x0c, y0c), (x0c, y1c)),
            ((x1c, y0c), (x1c, y1c)),
        ]
        for p0, p1 in segs:
            if p0 > p1:
                p0, p1 = p1, p0
            segments.add((p0, p1))

    for (p0, p1) in segments:
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            color=rect_color,
            linewidth=rect_lw,
            solid_capstyle="butt",
            solid_joinstyle="miter",
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def plot_multires_summary(
    rows: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]]]],
    out_path: Path,
    show: bool = False,
) -> None:
    """
    rows: (label, rgb, freq_map, scores, ys, xs, topk)
    """
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(16, 5 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, (label, rgb, freq_map, scores, ys, xs, topk) in enumerate(rows):
        ax0, ax1, ax2 = axes[i, 0], axes[i, 1], axes[i, 2]

        # (1) overlay
        ax0.imshow(rgb)
        ax0.set_title(f"{label} | overlay")
        ax0.axis("off")

        h0, w0 = rgb.shape[:2]

        def _clamp_x0(x: int) -> int:
            return int(max(0, min(w0 - 1, x)))

        def _clamp_y0(y: int) -> int:
            return int(max(0, min(h0 - 1, y)))

        segments = set()
        for (r, c) in topk:
            y0, y1 = int(ys[r]), int(ys[r + 1])
            x0, x1 = int(xs[c]), int(xs[c + 1])
            x0c, x1c = _clamp_x0(x0), _clamp_x0(x1)
            y0c, y1c = _clamp_y0(y0), _clamp_y0(y1)
            segs = [
                ((x0c, y0c), (x1c, y0c)),
                ((x0c, y1c), (x1c, y1c)),
                ((x0c, y0c), (x0c, y1c)),
                ((x1c, y0c), (x1c, y1c)),
            ]
            for p0, p1 in segs:
                if p0 > p1:
                    p0, p1 = p1, p0
                segments.add((p0, p1))

        for (p0, p1) in segments:
            ax0.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                color="red",
                linewidth=2.0,
                solid_capstyle="butt",
                solid_joinstyle="miter",
            )

        # (2) frequency map
        ax1.imshow(_normalize01(freq_map), cmap="gray")
        ax1.set_title(f"{label} | M^h (resized)")
        ax1.axis("off")

        # (3) grid scores
        im = ax2.imshow(_normalize01(scores), cmap="viridis")
        ax2.set_title(f"{label} | grid scores (mean)")
        ax2.set_xticks([])
        ax2.set_yticks([])
        for (r, c) in topk:
            ax2.scatter([c], [r], s=40, c="red", marker="x")
        fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def _compressed_display_px(
    max_side: int,
    min_side: int,
    max_side_all: int,
    cell_px: int,
    min_ratio: float = 0.70,
    max_ratio: float = 0.95,
) -> int:
    """
    把不同分辨率映射到“显示像素大小”，用于九宫格里体现分辨率差异，但不按原比例夸张缩放。

    - min_side -> cell_px * min_ratio
    - max_side_all -> cell_px * max_ratio
    - 中间用 log2 线性插值
    """
    if cell_px <= 0:
        raise ValueError("cell_px 必须 > 0")
    if min_side <= 0 or max_side_all <= 0 or max_side <= 0:
        return int(round(cell_px * max_ratio))
    if max_side_all == min_side:
        t = 1.0
    else:
        # log2 插值更符合“分辨率倍数”的感知
        t = float(np.log2(max_side / min_side) / np.log2(max_side_all / min_side))
        t = max(0.0, min(1.0, t))
    px = int(round(cell_px * (min_ratio + t * (max_ratio - min_ratio))))
    return max(1, min(cell_px, px))


def _render_overlay_pil(
    rgb: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    topk: List[Tuple[int, int]],
    rect_color: Tuple[int, int, int] = (255, 0, 0),
    rect_width: int = 4,
) -> Image.Image:
    """
    用 PIL 直接画 overlay（比 matplotlib 更“干净”，便于拼九宫格）。
    """
    # Pillow 未来版本会移除 mode 参数，这里让其自动推断
    img = Image.fromarray(rgb.astype(np.uint8))
    draw = ImageDraw.Draw(img)

    # 先把所有边“线段”去重，再统一画线，避免重叠边导致变粗
    h, w = rgb.shape[:2]

    def _clamp_x(x: int) -> int:
        return int(max(0, min(w - 1, x)))

    def _clamp_y(y: int) -> int:
        return int(max(0, min(h - 1, y)))

    segments = set()
    for (r, c) in topk:
        y0, y1 = int(ys[r]), int(ys[r + 1])
        x0, x1 = int(xs[c]), int(xs[c + 1])

        x0c, x1c = _clamp_x(x0), _clamp_x(x1)
        y0c, y1c = _clamp_y(y0), _clamp_y(y1)

        segs = [
            ((x0c, y0c), (x1c, y0c)),  # top
            ((x0c, y1c), (x1c, y1c)),  # bottom
            ((x0c, y0c), (x0c, y1c)),  # left
            ((x1c, y0c), (x1c, y1c)),  # right
        ]
        for p0, p1 in segs:
            # 统一方向，便于去重
            if p0 > p1:
                p0, p1 = p1, p0
            segments.add((p0, p1))

    lw = max(1, int(rect_width))
    for (p0, p1) in segments:
        draw.line([p0, p1], fill=rect_color, width=lw)
    return img


def montage9(
    image_path: Path,
    sizes: List[SizeSpec],
    grids: Optional[List[int]],
    windows: Optional[List[int]],
    topk: int,
    out_dir: Path,
    freq_source: str,
    freq_image: Optional[Path],
    rectify: str,
    show: bool = False,
    cell_px: int = 520,
    style: str = "debug",  # debug: 每格写明参数；paper: (a)-(i) 面板标注
    label_font_size: int = 28,
    export_dir: Optional[Path] = None,
) -> Path:
    """
    生成 3×3 九宫格：行=3个尺寸，列=3个grid。
    每个格子里只放 overlay 图（选窗结果），并让不同分辨率在格子里“显示大小”略有差异。
    """
    if len(sizes) != 3:
        raise ValueError("--montage9 需要恰好 3 个 sizes（3 行）")
    if (grids is None) == (windows is None):
        raise ValueError("--montage9 需要提供 grids 或 windows（二选一）")
    if grids is not None and len(grids) != 3:
        raise ValueError("--montage9 需要恰好 3 个 grids（3 列）")
    if windows is not None and len(windows) != 3:
        raise ValueError("--montage9 需要恰好 3 个 windows（3 列）")

    if not image_path.exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")

    img_pil = Image.open(str(image_path)).convert("RGB")
    inferred_freq = _infer_frequency2_path(image_path)
    if freq_image is None and inferred_freq is not None:
        freq_image = inferred_freq

    freq_source = freq_source.lower()
    if freq_source not in {"auto", "dataset", "dhwt"}:
        raise ValueError("freq_source 只能是 auto/dataset/dhwt")

    freq_pil_dataset: Optional[Image.Image] = None
    if freq_source in {"auto", "dataset"} and freq_image is not None and freq_image.exists():
        freq_pil_dataset = Image.open(str(freq_image)).convert("L")

    resampling = _get_resampling()

    # 计算分辨率映射（用 max_side）
    sides = [max(s.h, s.w) for s in sizes]
    min_side = int(min(sides))
    max_side = int(max(sides))

    style = style.lower().strip()
    if style not in {"debug", "paper"}:
        raise ValueError("montage9 style 只能是 debug 或 paper")

    # 画布布局（paper 更紧凑，更适合论文排版）
    header_h = 28 if style == "debug" else 0
    gap = 24 if style == "debug" else 18
    margin = 30 if style == "debug" else 16
    cell_w = cell_px
    cell_h = header_h + cell_px
    canvas_w = margin * 2 + 3 * cell_w + 2 * gap
    canvas_h = margin * 2 + 3 * cell_h + 2 * gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    font_debug = ImageFont.load_default()
    font_label = _load_times_new_roman(label_font_size)

    panel_labels = [f"({chr(ord('a') + i)})" for i in range(9)]

    # 可选：导出每个子图（不带字母/不带标题），便于论文排版
    if export_dir is not None:
        export_dir.mkdir(parents=True, exist_ok=True)

    # 逐格生成 overlay 并贴图
    for r, spec in enumerate(sizes):
        disp_px = _compressed_display_px(
            max_side=max(spec.h, spec.w),
            min_side=min_side,
            max_side_all=max_side,
            cell_px=cell_px,
        )
        col_list = grids if grids is not None else windows
        assert col_list is not None
        for c, col_val in enumerate(col_list):
            h, w = spec.h, spec.w
            label = spec.label

            # 直接 resize 到 H×W（不保持比例）
            img_resized = _resize_pil(img_pil, (h, w), resample=resampling.BICUBIC)
            rgb = np.array(img_resized)

            # 频率图 resize 到 H×W
            if freq_source == "dhwt":
                mh_half = frequency_map_from_dhwt(rgb, rectify=rectify)
                mh = resize_frequency_map_to_hw(mh_half, (h, w))
            else:
                if freq_pil_dataset is None:
                    if freq_source == "dataset":
                        raise FileNotFoundError(
                            "freq_source=dataset 但未找到频率图。"
                            "请提供 --freq_image，或保证 Image 同级存在 Frequency_2/同名文件。"
                        )
                    mh_half = frequency_map_from_dhwt(rgb, rectify=rectify)
                    mh = resize_frequency_map_to_hw(mh_half, (h, w))
                else:
                    mh_img = _resize_pil(freq_pil_dataset, (h, w), resample=resampling.BILINEAR)
                    mh = np.array(mh_img, dtype=np.float32)

            if windows is not None:
                scores, ys, xs = window_scores(mh, window=int(col_val))
                topk_list = topk_cells(scores, k=topk)
                header = f"{label} | win={int(col_val)} | grid={scores.shape[0]}x{scores.shape[1]}"
            else:
                g = int(col_val)
                scores, ys, xs = grid_scores(mh, grid=g)
                topk_list = topk_cells(scores, k=topk)
                win_h = h / float(g)
                win_w = w / float(g)
                header = f"{label} | g{g} | win≈{win_h:.0f}x{win_w:.0f}"
            overlay = _render_overlay_pil(rgb, ys, xs, topk_list, rect_color=(255, 0, 0), rect_width=4)
            overlay_disp = overlay.resize((disp_px, disp_px), resample=resampling.LANCZOS)

            if export_dir is not None:
                # 导出“干净”版本：仅图片本身（不含字母/不含参数文字），文件名=resolution + grid/window size
                res_name = f"{spec.h}x{spec.w}"
                if windows is not None:
                    gs_name = f"win{int(col_val)}"
                else:
                    gs_name = f"g{int(col_val)}"
                # 统一给一个白底画布，保证尺寸一致、方便拼版
                panel = Image.new("RGB", (cell_px, cell_px), color=(255, 255, 255))
                px0 = (cell_px - disp_px) // 2
                py0 = (cell_px - disp_px) // 2
                panel.paste(overlay_disp, (px0, py0))
                panel.save(str(export_dir / f"{res_name}_{gs_name}.png"))

            # 计算 cell 左上角
            x0 = margin + c * (cell_w + gap)
            y0 = margin + r * (cell_h + gap)

            # image area 内居中贴图
            img_area_x0 = x0
            img_area_y0 = y0 + header_h
            px = img_area_x0 + (cell_px - disp_px) // 2
            py = img_area_y0 + (cell_px - disp_px) // 2
            canvas.paste(overlay_disp, (px, py))

            if style == "debug":
                # header 文本
                draw.text((x0 + 4, y0 + 4), header, fill=(0, 0, 0), font=font_debug)
                # cell 边框（轻微）
                draw.rectangle([x0, y0, x0 + cell_w, y0 + cell_h], outline=(220, 220, 220), width=1)
            else:
                # 论文风格：仅标注 (a)-(i)，其余说明放到 caption
                idx = r * 3 + c
                lab = panel_labels[idx]
                # 在左上角加白底，保证可读性
                pad = 4
                try:
                    bbox = font_label.getbbox(lab)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                except Exception:
                    tw, th = (30, 18)
                lx = x0 + 8
                ly = y0 + 8
                draw.rectangle([lx - pad, ly - pad, lx + tw + pad, ly + th + pad], fill=(255, 255, 255))
                draw.text((lx, ly), lab, fill=(0, 0, 0), font=font_label)

    out_dir.mkdir(parents=True, exist_ok=True)
    tag_sizes = "-".join([s.label for s in sizes])
    if windows is not None:
        tag_cols = "-".join([str(w) for w in windows])
        suffix = "paper" if style == "paper" else "debug"
        out_path = out_dir / f"{image_path.stem}_montage9_{suffix}_sizes{tag_sizes}_wins{tag_cols}_top{topk}_freq{freq_source}.png"
    else:
        assert grids is not None
        tag_cols = "-".join([str(g) for g in grids])
        suffix = "paper" if style == "paper" else "debug"
        out_path = out_dir / f"{image_path.stem}_montage9_{suffix}_sizes{tag_sizes}_grids{tag_cols}_top{topk}_freq{freq_source}.png"
    canvas.save(str(out_path))
    if show:
        canvas.show()
    return out_path


def run(
    image_path: Path,
    sizes: List[SizeSpec],
    grid: int,
    window: Optional[int],
    topk: int,
    out_dir: Path,
    freq_source: str = "auto",
    freq_image: Optional[Path] = None,
    rectify: str = "relu",
    show: bool = False,
) -> List[Path]:
    if not image_path.exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")

    img_pil = Image.open(str(image_path)).convert("RGB")
    orig_hw = (img_pil.size[1], img_pil.size[0])  # (H,W)

    # 频率图（dataset）路径准备：只在需要时使用
    inferred_freq = _infer_frequency2_path(image_path)
    if freq_image is None and inferred_freq is not None:
        freq_image = inferred_freq

    freq_source = freq_source.lower()
    if freq_source not in {"auto", "dataset", "dhwt"}:
        raise ValueError("freq_source 只能是 auto/dataset/dhwt")

    freq_pil_dataset: Optional[Image.Image] = None
    if freq_source in {"auto", "dataset"} and freq_image is not None and freq_image.exists():
        freq_pil_dataset = Image.open(str(freq_image)).convert("L")

    resampling = _get_resampling()
    outputs: List[Path] = []
    rows = []

    for spec in sizes:
        h, w = spec.h, spec.w
        label = spec.label

        # 直接 resize 到 H×W（不保持比例）
        img_resized = _resize_pil(img_pil, (h, w), resample=resampling.BICUBIC)
        rgb = np.array(img_resized)

        # 生成/读取频率图，并 resize 到 H×W
        if freq_source == "dhwt":
            mh_half = frequency_map_from_dhwt(rgb, rectify=rectify)
            mh = resize_frequency_map_to_hw(mh_half, (h, w))
        else:
            # auto/dataset：优先用 Frequency_2
            if freq_pil_dataset is None:
                if freq_source == "dataset":
                    raise FileNotFoundError(
                        "freq_source=dataset 但未找到频率图。"
                        "请提供 --freq_image，或保证 Image 同级存在 Frequency_2/同名文件。"
                    )
                # auto 兜底：用 dhwt 生成
                mh_half = frequency_map_from_dhwt(rgb, rectify=rectify)
                mh = resize_frequency_map_to_hw(mh_half, (h, w))
            else:
                mh_img = _resize_pil(freq_pil_dataset, (h, w), resample=resampling.BILINEAR)
                mh = np.array(mh_img, dtype=np.float32)

        if window is not None:
            scores, ys, xs = window_scores(mh, window=int(window))
            grid_tag = f"win{int(window)}"
        else:
            scores, ys, xs = grid_scores(mh, grid=grid)
            grid_tag = f"g{grid}"
        topk_list = topk_cells(scores, k=topk)

        if window is not None:
            title = (
                f"{image_path.name} | {label} | img={h}x{w} | win={int(window)} | "
                f"grid={scores.shape[0]}x{scores.shape[1]} | topk={topk} | freq={freq_source}"
            )
        else:
            title = f"{image_path.name} | {label} | img={h}x{w} | grid={grid} | topk={topk} | freq={freq_source}"
        out_path = out_dir / f"{image_path.stem}_freqwin_{label}_img{h}x{w}_{grid_tag}_top{topk}.png"
        plot_overlay_only(
            rgb=rgb,
            ys=ys,
            xs=xs,
            topk=topk_list,
            title=title,
            save_path=out_path,
            rect_color="red",
            rect_lw=2.6,
            show=show,
        )

        outputs.append(out_path)
        rows.append((label, rgb, mh, scores, ys, xs, topk_list))

    summary_path = out_dir / f"{image_path.stem}_freqwin_multires_{grid_tag}_top{topk}.png"
    plot_multires_summary(rows=rows, out_path=summary_path, show=show)
    outputs.append(summary_path)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="频域信息 grid 可视化（覆盖到原图上，论文逻辑：DHWT + top-k windows）")
    parser.add_argument("--image", type=str, required=True, help="输入图片路径（jpg/png 等）")
    parser.add_argument(
        "--sizes",
        type=str,
        default="512",
        help="目标尺寸列表（逗号分隔）：'512,1024' 或 'orig,512' 或 '512x384'；整数表示 H=W",
    )
    parser.add_argument("--grid", type=int, default=8, help="grid 划分大小（NxN）。512x512 + grid=8 等价于 window=64（单值）")
    parser.add_argument(
        "--grids",
        type=str,
        default="",
        help="可选：多个 grid 值（逗号分隔）用于 sweep，例如 '4,8,16,32'；若提供则覆盖 --grid",
    )
    parser.add_argument("--window", type=int, default=0, help="固定窗口像素大小（例如 64）。若 >0，则使用 window 模式而不是 grid 模式。")
    parser.add_argument(
        "--windows",
        type=str,
        default="",
        help="可选：多个 window 像素大小用于 sweep，例如 '64,32,16'；若提供则覆盖 --window",
    )
    parser.add_argument("--topk", type=int, default=10, help="选 top-k 窗口（覆盖画到原图上）")
    parser.add_argument(
        "--montage9",
        type=int,
        default=0,
        help="是否额外生成 3×3 九宫格汇总图（需要恰好 3 个 sizes 和 3 个 grids）",
    )
    parser.add_argument(
        "--montage9_style",
        type=str,
        default="debug",
        choices=["debug", "paper"],
        help="九宫格样式：debug(每格带参数说明) / paper(仅(a)-(i)面板标注，说明放caption)",
    )
    parser.add_argument(
        "--montage9_export_dir",
        type=str,
        default="",
        help="可选：导出九宫格的9张单图到该目录（不带字母/不带标题），文件名=resolution + grid/window size",
    )
    parser.add_argument(
        "--freq_source",
        type=str,
        default="auto",
        choices=["auto", "dataset", "dhwt"],
        help="频率图来源：auto(优先Frequency_2否则dhwt)/dataset(强制用Frequency_2)/dhwt(从当前分辨率图计算)",
    )
    parser.add_argument(
        "--freq_image",
        type=str,
        default="",
        help="可选：显式指定 Frequency_2 频率图路径（dataset/auto 会用）",
    )
    parser.add_argument(
        "--rectify",
        type=str,
        default="relu",
        choices=["none", "relu", "abs"],
        help="dhwt 生成的频率图后处理：none/relu/abs。默认 relu 更接近 cv2.imwrite 的负值裁剪效果",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path.cwd() / "frequency_grid_visual"),
        help="输出目录",
    )
    parser.add_argument("--show", type=int, default=0, help="是否弹窗显示（1=显示，0=只保存）")

    args = parser.parse_args()
    image_path = Path(args.image).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    show = bool(int(args.show))

    img_pil = Image.open(str(image_path)).convert("RGB")
    orig_hw = (img_pil.size[1], img_pil.size[0])
    sizes = _parse_sizes(args.sizes, orig_hw=orig_hw)

    freq_image = Path(args.freq_image).expanduser().resolve() if args.freq_image else None

    if args.grids.strip():
        grid_list = _parse_int_list(args.grids)
    else:
        grid_list = [int(args.grid)]

    if args.windows.strip():
        window_list = _parse_int_list(args.windows)
    elif int(args.window) > 0:
        window_list = [int(args.window)]
    else:
        window_list = []

    outputs: List[Path] = []
    if window_list:
        for w in window_list:
            outputs.extend(
                run(
                    image_path=image_path,
                    sizes=sizes,
                    grid=int(args.grid),
                    window=w,
                    topk=int(args.topk),
                    out_dir=out_dir,
                    freq_source=str(args.freq_source),
                    freq_image=freq_image,
                    rectify=str(args.rectify),
                    show=show,
                )
            )
    else:
        for g in grid_list:
            outputs.extend(
                run(
                    image_path=image_path,
                    sizes=sizes,
                    grid=g,
                    window=None,
                    topk=int(args.topk),
                    out_dir=out_dir,
                    freq_source=str(args.freq_source),
                    freq_image=freq_image,
                    rectify=str(args.rectify),
                    show=show,
                )
            )

    # 额外生成九宫格（只在 sizes×grids==9 时启用）
    if bool(int(args.montage9)):
        if len(sizes) != 3:
            raise ValueError("--montage9 需要恰好 3 个 sizes")
        if window_list:
            if len(window_list) != 3:
                raise ValueError("--montage9 + windows 模式需要恰好 3 个 windows（3 列）")
            m9 = montage9(
                image_path=image_path,
                sizes=sizes,
                grids=None,
                windows=window_list,
                topk=int(args.topk),
                out_dir=out_dir,
                freq_source=str(args.freq_source),
                freq_image=freq_image,
                rectify=str(args.rectify),
                show=show,
                style=str(args.montage9_style),
                export_dir=Path(args.montage9_export_dir).expanduser().resolve() if args.montage9_export_dir else None,
            )
        else:
            if len(grid_list) != 3:
                raise ValueError("--montage9 + grids 模式需要恰好 3 个 grids（3 列）")
            m9 = montage9(
                image_path=image_path,
                sizes=sizes,
                grids=grid_list,
                windows=None,
                topk=int(args.topk),
                out_dir=out_dir,
                freq_source=str(args.freq_source),
                freq_image=freq_image,
                rectify=str(args.rectify),
                show=show,
                style=str(args.montage9_style),
                export_dir=Path(args.montage9_export_dir).expanduser().resolve() if args.montage9_export_dir else None,
            )
        outputs.append(m9)
    print("Saved:")
    for p in outputs:
        print(str(p))


if __name__ == "__main__":
    main()
