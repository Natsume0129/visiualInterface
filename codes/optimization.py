import os
import json
import numpy as np
import cv2
import pywt
from PIL import Image, ImageOps

# ================== PATH SETTINGS ==================
GRAY_PATH = r"E:\京大\visual_interface\lena_experiment\picture\lena_original_gray.png"
SKETCH_PATH = r"E:\京大\visual_interface\lena_experiment\picture\lena_sketch_2.jpg"
OUT_ROOT = r"E:\京大\visual_interface\lena_experiment\plots_optimization_2"

OPTIMIZE_METRIC = "f1"   # "f1" | "precision" | "recall"
TOLERANCE_PX = 2
# ===================================================

# ================== SEARCH SPACE ==================
WAVELET_LIST = ["haar", "db2", "db4", "sym2", "sym4"]
LEVELS_LIST = [3, 4]

CONSISTENCY_K_LIST = [2, 3]
CONSISTENCY_Q_LIST = [0.85, 0.90, 0.92]

THRESH_MODE_LIST = ["otsu", "quantile"]
EDGE_Q_LIST = [0.90, 0.92]

MORPH_OPEN_K_LIST = [0, 3]
FUSION_LIST = ["l2", "max"]
# ===================================================


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def safe_imwrite(path, img):
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    buf.tofile(path)


def load_gray(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return np.array(img.convert("L"), dtype=np.uint8)


def pad_1d(x, level):
    m = 2 ** level
    pad = (m - len(x) % m) % m
    return np.pad(x, (0, pad), mode="edge"), pad


def swt_detail_abs_1d(x, wavelet, level):
    x, pad = pad_1d(x.astype(np.float32), level)
    coeffs = pywt.swt(x, wavelet, level=level)
    out = []
    for cA, cD in coeffs:
        d = np.abs(cD)
        if pad > 0:
            d = d[:-pad]
        out.append(d)
    return out


def wavelet_response_xy(gray, wavelet, levels):
    H, W = gray.shape
    Ex = [np.zeros((H, W), np.float32) for _ in range(levels)]
    Ey = [np.zeros((H, W), np.float32) for _ in range(levels)]

    for r in range(H):
        ds = swt_detail_abs_1d(gray[r], wavelet, levels)
        for l in range(levels):
            Ex[l][r] = ds[l]

    for c in range(W):
        ds = swt_detail_abs_1d(gray[:, c], wavelet, levels)
        for l in range(levels):
            Ey[l][:, c] = ds[l]

    return Ex, Ey


def normalize_u8(x):
    lo, hi = np.percentile(x, [1, 99])
    if hi <= lo:
        return np.zeros_like(x, np.uint8)
    return ((np.clip(x, lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)


def make_gt(sketch, shape):
    if sketch.shape != shape:
        sketch = cv2.resize(sketch, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    _, bw = cv2.threshold(
        cv2.GaussianBlur(sketch, (5, 5), 0),
        0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    if (bw > 0).mean() > 0.5:
        bw = 255 - bw
    return bw


def fuse(Ex, Ey, fusion):
    levels = len(Ex)
    w = np.array([2 ** i for i in range(levels)], np.float32)
    w /= w.sum()
    Exf = sum(w[i] * Ex[i] for i in range(levels))
    Eyf = sum(w[i] * Ey[i] for i in range(levels))

    if fusion == "l2":
        return np.sqrt(Exf ** 2 + Eyf ** 2)
    return np.maximum(np.abs(Exf), np.abs(Eyf))


def binarize(E, mode, q, morph):
    if mode == "otsu":
        _, b = cv2.threshold(E, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        t = np.quantile(E, q)
        b = (E >= t).astype(np.uint8) * 255

    if morph > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph))
        b = cv2.morphologyEx(b, cv2.MORPH_OPEN, k)

    return b


def consistency_mask(Ex, Ey, k, q):
    H, W = Ex[0].shape
    cnt = np.zeros((H, W), np.int32)
    for l in range(len(Ex)):
        S = np.maximum(Ex[l], Ey[l])
        thr = np.quantile(S, q)
        cnt += (S >= thr)
    return (cnt >= k).astype(np.uint8) * 255


def eval_prf(pred, gt):
    pred = pred > 0
    gt = gt > 0
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * TOLERANCE_PX + 1, 2 * TOLERANCE_PX + 1)
    )
    gt_d = cv2.dilate(gt.astype(np.uint8), k)

    tp = np.sum(pred & gt_d)
    fp = np.sum(pred & ~gt_d)
    fn = np.sum(~pred & gt)

    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)
    f = 2 * p * r / (p + r + 1e-9)
    return p, r, f


def score(p, r, f):
    return {"precision": p, "recall": r, "f1": f}[OPTIMIZE_METRIC]


def overlay(gray, gt, pred):
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img[gt > 0] = (0, 255, 0)
    img[pred > 0] = (0, 0, 255)
    img[(gt > 0) & (pred > 0)] = (0, 255, 255)
    return img


def main():
    ensure_dir(OUT_ROOT)
    gray = load_gray(GRAY_PATH)
    gt = make_gt(load_gray(SKETCH_PATH), gray.shape)

    best_score = None
    best_pack = None
    best_imgs = None

    for wavelet in WAVELET_LIST:
        for levels in LEVELS_LIST:
            Ex, Ey = wavelet_response_xy(gray, wavelet, levels)

            for fusion in FUSION_LIST:
                E = fuse(Ex, Ey, fusion)
                E_u8 = normalize_u8(E)

                for mode in THRESH_MODE_LIST:
                    qs = EDGE_Q_LIST if mode == "quantile" else [None]
                    for q in qs:
                        for mk in MORPH_OPEN_K_LIST:
                            pred_b = binarize(E_u8, mode, q, mk)
                            pb, rb, fb = eval_prf(pred_b, gt)

                            for ck in CONSISTENCY_K_LIST:
                                for cq in CONSISTENCY_Q_LIST:
                                    pred_c = consistency_mask(Ex, Ey, ck, cq)
                                    pc, rc, fc = eval_prf(pred_c, gt)

                                    s = score(pc, rc, fc)
                                    if best_score is None or s > best_score:
                                        best_score = s
                                        best_pack = {
                                            "wavelet": wavelet,
                                            "levels": levels,
                                            "fusion": fusion,
                                            "threshold_mode": mode,
                                            "edge_q": q,
                                            "morph_open_k": mk,
                                            "consistency_k": ck,
                                            "consistency_q": cq,
                                            "baseline": {"p": pb, "r": rb, "f1": fb},
                                            "consistency": {"p": pc, "r": rc, "f1": fc},
                                        }
                                        best_imgs = (pred_b, pred_c, E_u8)

    out = os.path.join(OUT_ROOT, "best_run")
    ensure_dir(out)

    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(best_pack, f, indent=2)

    safe_imwrite(os.path.join(out, "overlay_baseline.png"),
                 overlay(gray, gt, best_imgs[0]))
    safe_imwrite(os.path.join(out, "overlay_consistency.png"),
                 overlay(gray, gt, best_imgs[1]))
    safe_imwrite(os.path.join(out, "edge_energy.png"), best_imgs[2])
    safe_imwrite(os.path.join(out, "gt_bin.png"), gt)

    print("=== DONE ===")
    print("Best score:", best_score)
    print("Best params:", best_pack)
    print("Saved to:", out)


if __name__ == "__main__":
    main()
