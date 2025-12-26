##
# This script implements a wavelet-based edge detection pipeline.
# It processes a grayscale image to extract edges using multi-scale wavelet transforms.


import os
import numpy as np
import cv2
import pywt
from PIL import Image, ImageOps

# ================== EDIT HERE ==================
GRAY_PATH = r"E:\京大\visual_interface\lena_experiment\picture\lena_original_gray.png"
SKETCH_PATH = r"E:\京大\visual_interface\lena_experiment\picture\lena_sketch_2.jpg"
OUT_DIR = r"E:\京大\visual_interface\lena_experiment\plots_2"

# Wavelet settings
WAVELET = "haar"
LEVELS = 4  # number of scales

# Binarization / cleanup
USE_OTSU = True          # if False, uses MANUAL_THRESH in [0..255]
MANUAL_THRESH = 40
MORPH_OPEN_K = 3         # set 0 to disable
TOLERANCE_PX = 2         # dilation radius for tolerant matching

# Cross-scale consistency (the "A-b improvement")
CONSISTENCY_K = 2        # keep pixels significant in at least K scales
CONSISTENCY_Q = 0.90     # per-scale threshold by quantile (higher -> stricter)
# ==============================================


def ensure_dir(p: str) -> None:
    # Create output directory if it does not exist
    os.makedirs(p, exist_ok=True)


def safe_imwrite(path: str, img: np.ndarray) -> None:
    # Write images safely on Windows even if the path contains non-ASCII characters.
    # cv2.imwrite may silently fail (return False) for such paths.
    ext = os.path.splitext(path)[1].lower()
    if ext == "":
        ext = ".png"
        path = path + ext

    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"cv2.imencode failed: {path}")

    # Use tofile() to handle Unicode paths on Windows
    buf.tofile(path)

    if not os.path.exists(path):
        raise RuntimeError(f"Image was not written (path issue): {path}")


def load_gray_with_exif(path: str) -> np.ndarray:
    # Load image with EXIF orientation correction, then convert to 8-bit grayscale
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("L")
    return np.array(img, dtype=np.uint8)


def pad_to_pow2_len_1d(x: np.ndarray, level: int) -> tuple[np.ndarray, int]:
    # SWT requires length to be multiple of 2**level; pad by edge values if needed
    n = x.shape[0]
    m = 2 ** level
    pad_len = (m - (n % m)) % m
    if pad_len == 0:
        return x, 0
    x_pad = np.pad(x, (0, pad_len), mode="edge")
    return x_pad, pad_len


def swt_detail_abs_1d(x: np.ndarray, wavelet: str, level: int) -> list[np.ndarray]:
    # Stationary wavelet transform (no downsampling): returns same-length detail per level
    x_f = x.astype(np.float32)
    x_pad, pad_len = pad_to_pow2_len_1d(x_f, level)

    # Keep default return format: list of (cA, cD)
    coeffs = pywt.swt(x_pad, wavelet, level=level)

    details = []
    for item in coeffs:
        # Expected format: (cA, cD)
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            cD = item[1]
        else:
            # Fallback (in case a different format appears)
            cD = item
        d = np.abs(cD)
        if pad_len > 0:
            d = d[:-pad_len]
        details.append(d)

    # Ensure we return exactly `level` detail arrays
    return details[:level]


def wavelet_response_xy(gray: np.ndarray, wavelet: str, levels: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    # Compute multi-scale wavelet detail magnitude separately along x (rows) and y (cols)
    H, W = gray.shape

    Ex_levels = [np.zeros((H, W), dtype=np.float32) for _ in range(levels)]
    Ey_levels = [np.zeros((H, W), dtype=np.float32) for _ in range(levels)]

    # x-direction: process each row as a 1D signal
    for r in range(H):
        details = swt_detail_abs_1d(gray[r, :], wavelet, levels)
        for l in range(levels):
            Ex_levels[l][r, :] = details[l]

    # y-direction: process each column as a 1D signal
    for c in range(W):
        details = swt_detail_abs_1d(gray[:, c], wavelet, levels)
        for l in range(levels):
            Ey_levels[l][:, c] = details[l]

    return Ex_levels, Ey_levels


def normalize_to_uint8(x: np.ndarray) -> np.ndarray:
    # Robust normalization to 0..255 for visualization / thresholding
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    if hi <= lo:
        return np.zeros_like(x, dtype=np.uint8)
    y = (np.clip(x, lo, hi) - lo) / (hi - lo)
    return (y * 255.0 + 0.5).astype(np.uint8)


def make_gt_from_sketch(sketch: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    # Convert a hand-drawn sketch to a binary GT edge mask (edges=255, background=0)
    H, W = target_shape
    if sketch.shape != (H, W):
        # Use nearest neighbor to preserve line structure
        sketch = cv2.resize(sketch, (W, H), interpolation=cv2.INTER_NEAREST)

    # Otsu binarization (blur helps threshold stability)
    blur = cv2.GaussianBlur(sketch, (5, 5), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Auto-invert so that "edge lines" become white (255)
    # Heuristic: if most pixels are white, background is likely white => lines are black => invert
    white_ratio = (bw > 0).mean()
    if white_ratio > 0.5:
        bw = 255 - bw

    # Optional light cleanup: remove tiny speckles
    if MORPH_OPEN_K and MORPH_OPEN_K > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_OPEN_K, MORPH_OPEN_K))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)

    return bw


def binarize_edge_map(E_u8: np.ndarray) -> np.ndarray:
    # Convert edge strength map to a binary edge mask
    if USE_OTSU:
        _, b = cv2.threshold(E_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, b = cv2.threshold(E_u8, int(MANUAL_THRESH), 255, cv2.THRESH_BINARY)

    if MORPH_OPEN_K and MORPH_OPEN_K > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_OPEN_K, MORPH_OPEN_K))
        b = cv2.morphologyEx(b, cv2.MORPH_OPEN, k)

    return b


def consistency_mask(Ex_levels: list[np.ndarray], Ey_levels: list[np.ndarray], k: int, q: float) -> np.ndarray:
    # Keep pixels that are "significant" across multiple scales (scale-consistency)
    # Per level, threshold by quantile to avoid manual tuning across images.
    levels = len(Ex_levels)
    H, W = Ex_levels[0].shape

    count = np.zeros((H, W), dtype=np.int32)
    for l in range(levels):
        # Combine x/y for this scale (take max -> sharper, less blur than sum)
        S = np.maximum(Ex_levels[l], Ey_levels[l])
        thr = np.quantile(S, q)
        count += (S >= thr).astype(np.int32)

    mask = (count >= int(k)).astype(np.uint8) * 255
    return mask


def eval_prf(pred: np.ndarray, gt: np.ndarray, tolerance_px: int) -> tuple[float, float, float]:
    # Pixel-level Precision/Recall/F1 with a tolerance (dilate GT)
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)

    if tolerance_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * tolerance_px + 1, 2 * tolerance_px + 1)
        )
        gt_tol = cv2.dilate(gt_bin, k)
    else:
        gt_tol = gt_bin

    tp = int((pred_bin & gt_tol).sum())
    fp = int((pred_bin & (1 - gt_tol)).sum())
    fn = int(((1 - pred_bin) & gt_bin).sum())  # miss against strict GT

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return precision, recall, f1


def overlay_edges(gray: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    # Create an overlay image: GT edges in green, Pred edges in red, overlap appears yellow
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    gt_m = (gt > 0)
    pr_m = (pred > 0)
    base[gt_m] = (0, 255, 0)           # green (GT)
    base[pr_m] = (0, 0, 255)           # red (Pred)
    base[gt_m & pr_m] = (0, 255, 255)  # yellow (overlap)
    return base


def main():
    ensure_dir(OUT_DIR)
    print("OUT_DIR =", OUT_DIR, "exists =", os.path.exists(OUT_DIR))

    gray = load_gray_with_exif(GRAY_PATH)
    sketch = load_gray_with_exif(SKETCH_PATH)

    # 1) Build GT binary edge mask from the hand sketch
    gt = make_gt_from_sketch(sketch, gray.shape)
    safe_imwrite(os.path.join(OUT_DIR, "gt_bin.png"), gt)

    # 2) Wavelet multi-scale responses in x/y
    Ex_levels, Ey_levels = wavelet_response_xy(gray, WAVELET, LEVELS)

    # Save per-scale visualizations (useful for the report)
    for i in range(LEVELS):
        ex_u8 = normalize_to_uint8(Ex_levels[i])
        ey_u8 = normalize_to_uint8(Ey_levels[i])
        safe_imwrite(os.path.join(OUT_DIR, f"Ex_scale_{i+1}.png"), ex_u8)
        safe_imwrite(os.path.join(OUT_DIR, f"Ey_scale_{i+1}.png"), ey_u8)

    # 3) Baseline fusion across scales and directions
    # Weight coarser scales slightly more (higher i => coarser in SWT convention)
    weights = np.array([2 ** i for i in range(LEVELS)], dtype=np.float32)
    weights = weights / (weights.sum() + 1e-9)

    Ex = np.zeros_like(Ex_levels[0], dtype=np.float32)
    Ey = np.zeros_like(Ey_levels[0], dtype=np.float32)
    for i in range(LEVELS):
        Ex += weights[i] * Ex_levels[i]
        Ey += weights[i] * Ey_levels[i]

    # L2-like fusion (baseline)
    E = np.sqrt(Ex * Ex + Ey * Ey)
    E_u8 = normalize_to_uint8(E)
    safe_imwrite(os.path.join(OUT_DIR, "edge_energy.png"), E_u8)

    pred_bin = binarize_edge_map(E_u8)
    safe_imwrite(os.path.join(OUT_DIR, "pred_bin_baseline.png"), pred_bin)

    # 4) Improvement: cross-scale consistency mask
    cons = consistency_mask(Ex_levels, Ey_levels, CONSISTENCY_K, CONSISTENCY_Q)
    safe_imwrite(os.path.join(OUT_DIR, "pred_bin_consistency.png"), cons)

    # 5) Evaluation (baseline + consistency)
    p0, r0, f0 = eval_prf(pred_bin, gt, TOLERANCE_PX)
    p1, r1, f1 = eval_prf(cons, gt, TOLERANCE_PX)

    # Save overlays for quick visual inspection
    ov0 = overlay_edges(gray, gt, pred_bin)
    ov1 = overlay_edges(gray, gt, cons)
    safe_imwrite(os.path.join(OUT_DIR, "overlay_baseline.png"), ov0)
    safe_imwrite(os.path.join(OUT_DIR, "overlay_consistency.png"), ov1)

    # Print metrics
    print("=== Evaluation (tolerance_px = {}) ===".format(TOLERANCE_PX))
    print("[Baseline]     Precision={:.4f} Recall={:.4f} F1={:.4f}".format(p0, r0, f0))
    print("[Consistency]  Precision={:.4f} Recall={:.4f} F1={:.4f}".format(p1, r1, f1))
    print("Outputs saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
