import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def _haar_dwt_level1(channel):
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


def frequency_map_from_dhwt(img_rgb, rectify="relu"):
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


def process_dataset(image_dir, output_dir, target_size=(512, 512), rectify="relu"):
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(image_dir)
                          if f.endswith('.jpg') or f.endswith('.png')])

    total_files = len(image_files)
    print(f"Processing {total_files} images, target_size={target_size}, rectify={rectify}...")

    for filename in tqdm(image_files):
        img_path = os.path.join(image_dir, filename)
        img = Image.open(img_path).convert("RGB")
        img = img.resize((target_size[1], target_size[0]), resample=Image.BILINEAR)
        rgb = np.array(img, dtype=np.uint8)

        mh = frequency_map_from_dhwt(rgb, rectify=rectify)

        mh_min = mh.min()
        mh_max = mh.max()
        if mh_max - mh_min < 1e-8:
            freq_map = np.zeros_like(mh, dtype=np.uint8)
        else:
            freq_map = ((mh - mh_min) / (mh_max - mh_min) * 255.0).astype(np.uint8)

        basename = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{basename}.jpg")
        Image.fromarray(freq_map).save(output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Frequency Map Generation for HFP-SAM (matching source code)")
    parser.add_argument("--image_dir", required=True, help="Path to input images")
    parser.add_argument("--output_dir", required=True, help="Path to save frequency maps")
    parser.add_argument("--target_size", type=int, nargs=2, default=[512, 512],
                        help="Target size for resizing (height width)")
    parser.add_argument("--rectify", type=str, default="relu", choices=["none", "relu", "abs"],
                        help="Rectification method for high-frequency subbands")
    args = parser.parse_args()

    process_dataset(args.image_dir, args.output_dir, tuple(args.target_size), args.rectify)
    print(f"Frequency maps generated successfully! Saved to: {args.output_dir}")
