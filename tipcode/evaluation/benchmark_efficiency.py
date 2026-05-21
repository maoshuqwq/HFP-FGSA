"""
Benchmark efficiency of HFP-SAM components (FGA / FPS / FVM).

This script reports:
- Params / Trainable Params (M)
- FLOPs (G) for FGA and FVM (if fvcore is available)
- End-to-end latency (ms/img) and a runtime breakdown
- Peak GPU memory (MB) on CUDA (if available)

It is designed to directly address reviewer requests on efficiency analysis.

Typical usage (run from repo root):
  python3 tipcode/benchmark_efficiency.py \
    --sam_ckpt tipcode/sam_vit_b_01ec64.pth \
    --image_size 512 \
    --device cuda \
    --warmup 30 --iters 200 \
    --out_csv efficiency_results.csv \
    --out_tex efficiency_table.tex

Notes:
- The benchmark uses batch=1 and fixed resolution (default 512^2), matching the paper setup.
- FLOPs are reported for the *incremental modules* (FGA adapters / FVM head).
  For FPS, we report runtime only (algorithmic selection).
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import torch
    import torch.nn as nn
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "This benchmark requires PyTorch. Please run it in your training environment where torch is installed."
    ) from e

from segment_anything import sam_model_registry
from ..adapters.frequency_final_point import frequency_grid_mask, point_prompt
from ..adapters.frequency_adapter import fre_adapter

try:
    from fvcore.nn import flop_count
except Exception:  # pragma: no cover
    flop_count = None  # type: ignore[assignment]

# Silence verbose FLOPs-analysis logs (fvcore may warn about unsupported ops).
logging.getLogger("fvcore").setLevel(logging.ERROR)
logging.getLogger("fvcore.nn.jit_analysis").setLevel(logging.ERROR)


class NullFreAdapter(nn.Module):
    def forward(self, x: torch.Tensor, fre_mask: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return x


@dataclass
class BenchRow:
    variant: str
    image_size: int
    amp: bool
    params_m: float
    trainable_params_m: float
    fga_params_m: float
    fvm_params_m: float
    fga_flops_g: float
    fvm_flops_g: float
    latency_ms: float
    peak_mem_mb: float
    t_preprocess_ms: float
    t_fre_mask_ms: float
    t_image_encoder_ms: float
    t_decode1_ms: float
    t_fps_ms: float
    t_decode2_ms: float
    t_fvm_ms: float


def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _count_trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _device_from_arg(arg: str) -> torch.device:
    arg = arg.lower()
    if arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(arg)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _peak_mem_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return float("nan")
    return float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))


def _reset_peak_mem(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def _now() -> float:
    return time.perf_counter()


def _ms(dt_s: float) -> float:
    return 1000.0 * dt_s


def _extract_masks(out) -> torch.Tensor:
    # Our MaskDecoder returns (masks, iou_pred, new_mask). Keep robust to other layouts.
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def _fvcore_flops_g(model: nn.Module, inputs: Tuple[torch.Tensor, ...], supported_ops=None) -> float:
    if flop_count is None:
        return float("nan")
    flops_dict, _unsupported = flop_count(model=model, inputs=inputs, supported_ops=supported_ops)  # type: ignore[misc]
    # fvcore returns a dict in G-FLOPs
    return float(sum(flops_dict.values()))


def _selective_scan_flop_jit_no_print(inputs, outputs) -> int:  # noqa: ARG001
    """
    Custom FLOP handler for vmamba selective scan ops, adapted from vmamba.py but without prints.
    """
    # xs, dts, As, Bs, Cs, Ds (optional), z (optional), dt_projs_bias (optional)
    # shapes:
    #   xs: (B, D, L)
    #   As: (D, N)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    # Bs can be grouped: (B, G, N, L) or (B, N, L)
    with_group = len(inputs[3].type().sizes()) == 4
    # Ds present?
    with_d = False
    if len(inputs) >= 6:
        s5 = inputs[5].type().sizes()
        if s5 is not None and len(s5) == 1 and s5[0] == D:
            with_d = True
    # z is ignored here (rare for our usage); set with_Z=False
    from segment_anything.modeling.vmamba import flops_selective_scan_fn

    return int(flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=with_d, with_Z=False, with_Group=with_group))


def _supported_ops_vmamba() -> Dict[str, object]:
    # Match the keys used in vmamba.py for fvcore jit analysis.
    return {
        "aten::silu": None,
        "aten::neg": None,
        "aten::exp": None,
        "aten::flip": None,
        "prim::PythonOp.CrossScan": None,
        "prim::PythonOp.CrossMerge": None,
        "prim::PythonOp.SelectiveScan": _selective_scan_flop_jit_no_print,
        "prim::PythonOp.SelectiveScanFn": _selective_scan_flop_jit_no_print,
    }


@torch.no_grad()
def _run_once(
    sam: nn.Module,
    fre_adapters: nn.ModuleList,
    fre: torch.Tensor,
    box_t: torch.Tensor,
    use_fga: bool,
    use_fps: bool,
    enable_fvm: bool,
    device: torch.device,
    amp: bool,
    image_size: int,
) -> Tuple[Dict[str, float], float, float]:
    """
    One end-to-end pass with breakdown.
    Returns: (breakdown_ms, total_ms, peak_mem_mb)
    """
    breakdown: Dict[str, float] = {
        "preprocess": 0.0,
        "fre_mask": 0.0,
        "image_encoder": 0.0,
        "decode1": 0.0,
        "fps": 0.0,
        "decode2": 0.0,
        "fvm": 0.0,
    }

    # create random uint8-like image (float32 in [0,255])
    img = torch.randint(0, 256, (1, 3, image_size, image_size), device=device, dtype=torch.float32)

    _reset_peak_mem(device)
    _sync(device)
    t0 = _now()

    # preprocess
    t = _now()
    x = sam.preprocess(img)
    _sync(device)
    breakdown["preprocess"] += _ms(_now() - t)

    # FGA mask (computed from fre map)
    t = _now()
    if use_fga:
        fre_mask = frequency_grid_mask(fre.detach().cpu())  # CPU tensor (python slicing friendly)
    else:
        fre_mask = torch.zeros((fre.shape[0], 32, 32), dtype=torch.float32)
    _sync(device)
    breakdown["fre_mask"] += _ms(_now() - t)

    # image encoder
    t = _now()
    with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
        image_embeddings = sam.image_encoder(x, fre_mask, fre_adapters)
    _sync(device)
    breakdown["image_encoder"] += _ms(_now() - t)

    # ---- decode 1 (coarse; using full-image box prompt) ----
    # Disable FVM during coarse stage for a clean breakdown.
    if hasattr(sam.mask_decoder, "enable_desam"):
        sam.mask_decoder.enable_desam = False
    t = _now()
    with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
        sp1, de1 = sam.prompt_encoder(points=None, boxes=box_t, masks=None)
        out1 = sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sp1,
            dense_prompt_embeddings=de1,
            multimask_output=False,
        )
    low_res_1 = _extract_masks(out1)
    _sync(device)
    breakdown["decode1"] += _ms(_now() - t)

    # Optional refine stage (used by FPS/FVM variants).
    run_refine = bool(use_fps or enable_fvm)
    if run_refine:
        # Prepare mask prompt (low-res 256x256, binary argmax)
        mask_prompt = torch.argmax(torch.softmax(low_res_1, dim=1), dim=1, keepdim=True).to(dtype=torch.float32)

        # ---- FPS (point selection) ----
        points = None
        if use_fps:
            # Need a 512x512 coarse logit map for point selection
            t = _now()
            coarse_512 = sam.postprocess_masks(
                low_res_1, input_size=(image_size, image_size), original_size=(image_size, image_size)
            )
            c1 = coarse_512[:, 1, :, :].detach().cpu()
            grid_point, grid_value = point_prompt(fre.detach().cpu(), c1)
            points = (grid_point.to(device=device), grid_value.to(device=device))
            _sync(device)
            breakdown["fps"] += _ms(_now() - t)

        # ---- decode 2 (refine; mask prompt + optional FPS points) ----
        if hasattr(sam.mask_decoder, "enable_desam"):
            sam.mask_decoder.enable_desam = bool(enable_fvm)
        t = _now()
        with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
            sp2, de2 = sam.prompt_encoder(points=points, boxes=None, masks=mask_prompt)
            _ = sam.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sp2,
                dense_prompt_embeddings=de2,
                multimask_output=False,
            )
        _sync(device)
        breakdown["decode2"] += _ms(_now() - t)

    total_ms = _ms(_now() - t0)
    peak_mb = _peak_mem_mb(device)
    return breakdown, total_ms, peak_mb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam_ckpt", type=str, default=str(Path(__file__).parent / "sam_vit_b_01ec64.pth"))
    ap.add_argument("--fga_ckpt", type=str, default="", help="Optional checkpoint with fre_adapter weights (state_dict filter).")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu", "mps"])
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--amp", action="store_true", help="Use AMP (cuda only).")
    ap.add_argument(
        "--freeze_image_encoder",
        action="store_true",
        default=True,
        help="Freeze SAM image encoder when reporting trainable params (default: True, matching the paper).",
    )
    ap.add_argument(
        "--no_freeze_image_encoder",
        action="store_false",
        dest="freeze_image_encoder",
        help="Disable encoder freezing (not recommended for the paper tables).",
    )
    ap.add_argument("--out_csv", type=str, default="efficiency_results.csv")
    ap.add_argument("--out_tex", type=str, default="efficiency_table.tex")
    args = ap.parse_args()

    device = _device_from_arg(args.device)
    image_size = int(args.image_size)

    if flop_count is None:
        print("[WARN] fvcore is not available. FLOPs columns will be NaN. Install with: pip install fvcore")

    # Build SAM
    sam, _ = sam_model_registry["vit_b"](checkpoint=str(args.sam_ckpt))
    sam = sam.to(device)
    sam.eval()

    if args.freeze_image_encoder:
        for p in sam.image_encoder.parameters():
            p.requires_grad = False

    # Build FGA adapters
    def _build_adapters(enabled: bool) -> nn.ModuleList:
        if not enabled:
            return nn.ModuleList([NullFreAdapter() for _ in range(12)]).to(device)
        adapters = nn.ModuleList([fre_adapter(c=768, r=4) for _ in range(12)])
        if args.fga_ckpt:
            state = torch.load(args.fga_ckpt, map_location="cpu")
            sub = {k.replace("fre_adapter.", "", 1): v for k, v in state.items() if k.startswith("fre_adapter.")}
            adapters.load_state_dict(sub, strict=False)
        return adapters.to(device).eval()

    # Inputs (fre map is treated as precomputed frequency prior in [0,1])
    # Keep frequency maps on CPU for FPS/FGA mask selection (python slicing + top-k).
    fre = torch.rand((1, image_size, image_size), device="cpu", dtype=torch.float32)
    box_t = torch.tensor([[0.0, 0.0, float(image_size - 1), float(image_size - 1)]], device=device)

    variants = [
        ("SAM", dict(use_fga=False, use_fps=False, enable_fvm=False)),
        ("SAM+FGA", dict(use_fga=True, use_fps=False, enable_fvm=False)),
        ("SAM+FGA+FPS", dict(use_fga=True, use_fps=True, enable_fvm=False)),
        ("HFP-SAM (FGA+FPS+FVM)", dict(use_fga=True, use_fps=True, enable_fvm=True)),
    ]

    rows: List[BenchRow] = []

    # Split SAM params into "base SAM" and the optional FVM head for clearer reporting.
    fvm_params_full = _count_params(sam.mask_decoder.desam) if hasattr(sam.mask_decoder, "desam") else 0
    sam_params_full = _count_params(sam)
    sam_params_no_fvm = sam_params_full - fvm_params_full
    fvm_trainable_full = (
        _count_trainable_params(sam.mask_decoder.desam) if hasattr(sam.mask_decoder, "desam") else 0
    )
    sam_trainable_full = _count_trainable_params(sam)
    sam_trainable_no_fvm = sam_trainable_full - fvm_trainable_full

    for name, cfg in variants:
        use_fga = bool(cfg["use_fga"])
        use_fps = bool(cfg["use_fps"])
        enable_fvm = bool(cfg["enable_fvm"])

        fre_adapters = _build_adapters(enabled=use_fga)

        # Put FVM head on CPU for variants that do not use it, to reflect actual deployment cost.
        if hasattr(sam.mask_decoder, "desam"):
            sam.mask_decoder.desam = sam.mask_decoder.desam.to(device if enable_fvm else torch.device("cpu"))

        # params
        fga_params = _count_params(fre_adapters) if use_fga else 0
        fga_trainable = _count_trainable_params(fre_adapters) if use_fga else 0
        fvm_params = fvm_params_full if enable_fvm else 0
        fvm_trainable = fvm_trainable_full if enable_fvm else 0

        params_total = sam_params_no_fvm + fga_params + fvm_params
        trainable_total = sam_trainable_no_fvm + fga_trainable + fvm_trainable

        # FLOPs for incremental modules
        fga_flops_g = float("nan")
        fvm_flops_g = float("nan")
        if flop_count is not None:
            # FGA: stack of 12 adapters on a (1,32,32,768) feature map + (1,32,32) mask
            if use_fga:
                class _FGAStack(nn.Module):
                    def __init__(self, adapters: nn.ModuleList):
                        super().__init__()
                        self.adapters = adapters

                    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
                        for a in self.adapters:
                            x = a(x, m)
                        return x

                x0 = torch.randn((1, 32, 32, 768), device=device)
                m0 = torch.zeros((1, 32, 32), device=device)
                fga_flops_g = _fvcore_flops_g(_FGAStack(fre_adapters).to(device).eval(), (x0, m0))

            # FVM: desam head on (1,32,128,128)
            if hasattr(sam.mask_decoder, "desam"):
                desam = sam.mask_decoder.desam.to(device).eval()
                z0 = torch.randn((1, 32, 128, 128), device=device)
                fvm_flops_g = _fvcore_flops_g(desam, (z0,), supported_ops=_supported_ops_vmamba())

        # Only report FVM FLOPs for variants that actually enable it.
        fvm_flops_row = fvm_flops_g if enable_fvm else float("nan")

        # Warmup
        for _ in range(max(0, int(args.warmup))):
            _run_once(
                sam=sam,
                fre_adapters=fre_adapters,
                fre=fre,
                box_t=box_t,
                use_fga=use_fga,
                use_fps=use_fps,
                enable_fvm=enable_fvm,
                device=device,
                amp=bool(args.amp),
                image_size=image_size,
            )

        # Timed runs
        total_ms_list: List[float] = []
        peak_mb_list: List[float] = []
        agg = {k: 0.0 for k in ["preprocess", "fre_mask", "image_encoder", "decode1", "fps", "decode2", "fvm"]}
        n = max(1, int(args.iters))
        for _ in range(n):
            breakdown, total_ms, peak_mb = _run_once(
                sam=sam,
                fre_adapters=fre_adapters,
                fre=fre,
                box_t=box_t,
                use_fga=use_fga,
                use_fps=use_fps,
                enable_fvm=enable_fvm,
                device=device,
                amp=bool(args.amp),
                image_size=image_size,
            )
            total_ms_list.append(total_ms)
            peak_mb_list.append(peak_mb)
            for k, v in breakdown.items():
                agg[k] += float(v)

        latency_ms = float(sum(total_ms_list) / len(total_ms_list))
        peak_mb = float(max(peak_mb_list)) if device.type == "cuda" else float("nan")

        row = BenchRow(
            variant=name,
            image_size=image_size,
            amp=bool(args.amp),
            params_m=params_total / 1e6,
            trainable_params_m=trainable_total / 1e6,
            fga_params_m=fga_params / 1e6,
            fvm_params_m=fvm_params_full / 1e6,
            fga_flops_g=fga_flops_g,
            fvm_flops_g=fvm_flops_row,
            latency_ms=latency_ms,
            peak_mem_mb=peak_mb,
            t_preprocess_ms=agg["preprocess"] / n,
            t_fre_mask_ms=agg["fre_mask"] / n,
            t_image_encoder_ms=agg["image_encoder"] / n,
            t_decode1_ms=agg["decode1"] / n,
            t_fps_ms=agg["fps"] / n,
            t_decode2_ms=agg["decode2"] / n,
            t_fvm_ms=agg["fvm"] / n,
        )
        rows.append(row)
        print(f"[OK] {name}: latency={latency_ms:.3f} ms | peak_mem={peak_mb:.1f} MB | params={row.params_m:.2f} M")

    # Derive an isolated FVM runtime estimate from the two comparable variants.
    # (Decode2 with FVM) - (Decode2 without FVM).
    base_no_fvm = next((r for r in rows if r.variant == "SAM+FGA+FPS"), None)
    full_fvm = next((r for r in rows if r.variant == "HFP-SAM (FGA+FPS+FVM)"), None)
    if base_no_fvm is not None and full_fvm is not None:
        full_fvm.t_fvm_ms = max(0.0, float(full_fvm.t_decode2_ms - base_no_fvm.t_decode2_ms))

    # Write CSV
    out_csv = Path(args.out_csv).expanduser()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        wr.writeheader()
        wr.writerows([asdict(r) for r in rows])
    print("Wrote:", str(out_csv))

    # Write LaTeX snippet (small table for the paper)
    out_tex = Path(args.out_tex).expanduser()
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    with out_tex.open("w") as f:
        f.write("% Auto-generated by tipcode/benchmark_efficiency.py\n")
        # One compact table: runtime breakdown + (Latency, Peak Mem) moved from the removed efficiency table.
        f.write("\\begin{table}[t]\\centering\\small\n")
        f.write(
            "\\caption{Runtime breakdown (ms/img) and overall cost. "
            "FGA-mask corresponds to the frequency-guided mask generation; "
            "FPS is the point selection time; $\\Delta$FVM is derived as the decode2 difference between w/ and w/o FVM.}"
            "\\label{tab:runtime_breakdown}\n"
        )
        f.write("\\resizebox{0.48\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{l|c|c|c|c|c|c|c|c}\\hline\n")
        f.write(
            "\\textbf{Variant} & \\textbf{Enc} & \\textbf{Dec1} & \\textbf{FGA-mask} & \\textbf{FPS} & "
            "\\textbf{Dec2} & $\\mathbf{\\Delta}$\\textbf{FVM} & \\textbf{Latency} & \\textbf{Peak Mem}\\\\\\hline\n"
        )
        for r in rows:
            name_disp = "HFP-SAM" if r.variant.startswith("HFP-SAM") else r.variant
            peak = f"{r.peak_mem_mb:.0f}" if r.peak_mem_mb == r.peak_mem_mb else "--"
            f.write(
                f"{name_disp} & {r.t_image_encoder_ms:.2f} & {r.t_decode1_ms:.2f} & {r.t_fre_mask_ms:.2f} & "
                f"{r.t_fps_ms:.2f} & {r.t_decode2_ms:.2f} & {r.t_fvm_ms:.2f} & {r.latency_ms:.2f} & {peak}\\\\\n"
            )
        f.write("\\hline\\end{tabular}}\n")
        f.write("\\vspace{-3mm}\n")
        f.write("\\end{table}\n")
    print("Wrote:", str(out_tex))


if __name__ == "__main__":
    main()


