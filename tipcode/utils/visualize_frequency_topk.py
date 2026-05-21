import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

from generate_frequency_maps import frequency_map_from_dhwt


def sliding_window_np(freq_map, window_size):
    h, w = freq_map.shape
    n_rows = h // window_size
    n_cols = w // window_size
    h2 = n_rows * window_size
    w2 = n_cols * window_size
    crop = freq_map[:h2, :w2]
    windows = crop.reshape(n_rows, window_size, n_cols, window_size)
    sums = windows.sum(axis=(1, 3))
    means = windows.mean(axis=(1, 3))
    return sums, means


def top_k_windows_np(scores, k, window_size):
    flat = scores.reshape(-1)
    k = min(k, flat.size)
    idx = np.argpartition(flat, -k)[-k:]
    idx = idx[np.argsort(flat[idx])[::-1]]
    n_cols = scores.shape[1]
    y_indices = (idx // n_cols) * window_size
    x_indices = (idx % n_cols) * window_size
    return y_indices, x_indices


def draw_topk_on_image(img_rgb, y_indices, x_indices, window_size, scores=None):
    vis = img_rgb.copy()
    for i, (y, x) in enumerate(zip(y_indices, x_indices)):
        y, x = int(y), int(x)
        cv2.rectangle(vis, (x, y), (x + window_size, y + window_size), (0, 0, 255), 2)
        label = f"#{i+1}"
        if scores is not None:
            label += f" {scores[i]:.0f}"
        cv2.putText(vis, label, (x + 2, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
    return vis


def draw_heatmap_overlay(img_rgb, freq_map_uint8):
    h, w = img_rgb.shape[:2]
    freq_resized = cv2.resize(freq_map_uint8, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(freq_resized, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_rgb, 0.5, heatmap, 0.5, 0)
    return overlay


def draw_gt_contours(vis, gt_path, target_size):
    if gt_path is None or not os.path.exists(gt_path):
        return vis
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    gt_resized = cv2.resize(gt, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(gt_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 1)
    return vis


def find_gt_path(gt_dir, basename):
    if gt_dir is None:
        return None
    for ext in [".png", ".jpg"]:
        p = os.path.join(gt_dir, f"{basename}{ext}")
        if os.path.exists(p):
            return p
    return None


def process_dataset(image_dir, output_freq_dir, output_vis_dir,
                    gt_dir=None, target_size=(512, 512), k=5, window_size=32, rectify="relu"):
    freq_dir = output_freq_dir
    heatmap_dir = os.path.join(output_vis_dir, "heatmap")
    sum_dir = os.path.join(output_vis_dir, f"topk_sum_w{window_size}")
    mean_dir = os.path.join(output_vis_dir, f"topk_mean_w{window_size}")
    for d in [freq_dir, heatmap_dir, sum_dir, mean_dir]:
        os.makedirs(d, exist_ok=True)

    image_files = sorted([f for f in os.listdir(image_dir)
                          if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

    total = len(image_files)
    print(f"Processing {total} images")
    print(f"  target_size={target_size}, k={k}, window_size={window_size}, rectify={rectify}")
    print(f"  freq_dir:     {freq_dir}")
    print(f"  heatmap_dir:  {heatmap_dir}")
    print(f"  sum_vis_dir:  {sum_dir}")
    print(f"  mean_vis_dir: {mean_dir}")

    for filename in tqdm(image_files):
        img_path = os.path.join(image_dir, filename)
        img_pil = Image.open(img_path).convert("RGB")
        img_resized = img_pil.resize((target_size[1], target_size[0]), resample=Image.BILINEAR)
        rgb = np.array(img_resized, dtype=np.uint8)

        mh = frequency_map_from_dhwt(rgb, rectify=rectify)

        mh_min = mh.min()
        mh_max = mh.max()
        if mh_max - mh_min < 1e-8:
            freq_map_uint8 = np.zeros_like(mh, dtype=np.uint8)
        else:
            freq_map_uint8 = ((mh - mh_min) / (mh_max - mh_min) * 255.0).astype(np.uint8)

        basename = os.path.splitext(filename)[0]

        Image.fromarray(freq_map_uint8).save(os.path.join(freq_dir, f"{basename}.jpg"))

        heatmap_vis = draw_heatmap_overlay(rgb, freq_map_uint8)
        gt_path = find_gt_path(gt_dir, basename)
        heatmap_vis = draw_gt_contours(heatmap_vis, gt_path, target_size)
        cv2.imwrite(os.path.join(heatmap_dir, f"{basename}.png"),
                    cv2.cvtColor(heatmap_vis, cv2.COLOR_RGB2BGR))

        freq_float = freq_map_uint8.astype(np.float64)
        sums, means = sliding_window_np(freq_float, window_size)

        fh, fw = freq_float.shape
        sy, sx = top_k_windows_np(sums, k, window_size)
        s_scores = sums[sy // window_size, sx // window_size]
        sy_full = (sy.astype(np.float64) / fh * target_size[0]).astype(int)
        sx_full = (sx.astype(np.float64) / fw * target_size[1]).astype(int)
        ws_full = int(window_size / fh * target_size[0])
        sum_vis = draw_topk_on_image(rgb, sy_full, sx_full, ws_full, s_scores)
        sum_vis = draw_gt_contours(sum_vis, gt_path, target_size)
        cv2.imwrite(os.path.join(sum_dir, f"{basename}.png"),
                    cv2.cvtColor(sum_vis, cv2.COLOR_RGB2BGR))

        my, mx = top_k_windows_np(means, k, window_size)
        m_scores = means[my // window_size, mx // window_size]
        my_full = (my.astype(np.float64) / fh * target_size[0]).astype(int)
        mx_full = (mx.astype(np.float64) / fw * target_size[1]).astype(int)
        wm_full = int(window_size / fh * target_size[0])
        mean_vis = draw_topk_on_image(rgb, my_full, mx_full, wm_full, m_scores)
        mean_vis = draw_gt_contours(mean_vis, gt_path, target_size)
        cv2.imwrite(os.path.join(mean_dir, f"{basename}.png"),
                    cv2.cvtColor(mean_vis, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Visualize frequency top-k windows on original images")
    parser.add_argument("--image_dir", required=True, help="Path to input images")
    parser.add_argument("--output_freq_dir", required=True, help="Path to save frequency maps")
    parser.add_argument("--output_vis_dir", required=True, help="Path to save visualization root dir")
    parser.add_argument("--gt_dir", default=None, help="Path to GT masks (optional, draws green contours)")
    parser.add_argument("--target_size", type=int, nargs=2, default=[512, 512],
                        help="Target size for resizing (height width)")
    parser.add_argument("--k", type=int, default=5, help="Number of top-k windows to select")
    parser.add_argument("--window_size", type=int, default=32, help="Sliding window size")
    parser.add_argument("--rectify", type=str, default="relu", choices=["none", "relu", "abs"])
    args = parser.parse_args()

    process_dataset(
        args.image_dir, args.output_freq_dir, args.output_vis_dir,
        args.gt_dir, tuple(args.target_size), args.k, args.window_size, args.rectify,
    )
    print("Done!")
