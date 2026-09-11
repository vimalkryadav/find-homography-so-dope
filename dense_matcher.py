"""Dense correspondence via LoFTR, used only where sparse matching thins out.

Optional: solution_v15 falls back to the sparse path if torch/kornia are absent,
so the module stays importable on a machine without them.
"""
import cv2
import numpy as np

LONG_SIDE = 840
CONF_MIN = 0.5
RANSAC_THRESH = 3.0

_matcher = None


def _get_matcher():
    global _matcher
    if _matcher is None:
        import torch
        import kornia.feature as KF
        _matcher = KF.LoFTR(pretrained="outdoor").eval()
    return _matcher


def _load(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    import torch
    h, w = img.shape[:2]
    scale = LONG_SIDE / max(h, w)
    nw = max(8, int(round(w * scale / 8)) * 8)
    nh = max(8, int(round(h * scale / 8)) * 8)
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(small).float()[None, None] / 255.0, (w / nw, h / nh)


def dense_homography(path1, path2):
    import torch
    t1, s1 = _load(path1)
    t2, s2 = _load(path2)
    if t1 is None or t2 is None:
        return None
    with torch.inference_mode():
        out = _get_matcher()({"image0": t1, "image1": t2})
    k0 = out["keypoints0"].numpy()
    k1 = out["keypoints1"].numpy()
    conf = out["confidence"].numpy()
    keep = conf >= CONF_MIN
    k0, k1 = k0[keep] * np.array(s1), k1[keep] * np.array(s2)
    if len(k0) < 4:
        return None
    H, _ = cv2.findHomography(k0.reshape(-1, 1, 2).astype(np.float32),
                              k1.reshape(-1, 1, 2).astype(np.float32),
                              cv2.USAC_MAGSAC, RANSAC_THRESH,
                              maxIters=20000, confidence=0.9999)
    return H
