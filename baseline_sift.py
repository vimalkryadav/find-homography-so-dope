"""SIFT -> ratio-test matching -> MAGSAC++ homography.

Exposes the predict() entry point the official benchmark.py imports:

    def predict(image_1_path, image_2_path) -> np.ndarray   # 3x3, image 1 -> image 2

Never raises and never returns a singular matrix: a pair that cannot be matched
falls back to ORB and then to identity, because the scorer penalises a missing
or degenerate H far more than a merely inaccurate one.
"""
import cv2
import numpy as np

NFEATURES = 8000
RATIO = 0.75
RANSAC_THRESH = 3.0
CONFIDENCE = 0.9999
MAX_ITERS = 10000
MIN_MATCHES = 4

IDENTITY = np.eye(3, dtype=np.float64)

_sift = None
_orb = None


def _detectors():
    global _sift, _orb
    if _sift is None:
        _sift = cv2.SIFT_create(nfeatures=NFEATURES)
        _orb = cv2.ORB_create(nfeatures=NFEATURES)
    return _sift, _orb


def _load(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"could not read {path}")
    return img


def _ratio_matches(des1, des2, norm):
    """Lowe ratio test, run both directions and kept only where mutual."""
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return []
    bf = cv2.BFMatcher(norm)

    def one_way(a, b):
        good = {}
        for pair in bf.knnMatch(a, b, k=2):
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < RATIO * n.distance:
                good[m.queryIdx] = m.trainIdx
        return good

    fwd = one_way(des1, des2)
    rev = one_way(des2, des1)
    return [(q, t) for q, t in fwd.items() if rev.get(t) == q]


def _fit(kp1, kp2, matches):
    if len(matches) < MIN_MATCHES:
        return None
    src = np.float32([kp1[q].pt for q, _ in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[t].pt for _, t in matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(
        src, dst, cv2.USAC_MAGSAC,
        ransacReprojThreshold=RANSAC_THRESH,
        maxIters=MAX_ITERS, confidence=CONFIDENCE,
    )
    return _sanitise(H)


def _sanitise(H):
    """Reject unusable matrices and normalise so h33 == 1."""
    if H is None:
        return None
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3) or not np.all(np.isfinite(H)):
        return None
    if abs(H[2, 2]) < 1e-12:
        return None
    H = H / H[2, 2]
    if not np.all(np.isfinite(H)) or abs(float(np.linalg.det(H))) < 1e-12:
        return None
    return H


def predict(image_1_path: str, image_2_path: str) -> np.ndarray:
    try:
        img1, img2 = _load(image_1_path), _load(image_2_path)
        sift, orb = _detectors()

        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        H = _fit(kp1, kp2, _ratio_matches(des1, des2, cv2.NORM_L2))
        if H is not None:
            return H

        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        H = _fit(kp1, kp2, _ratio_matches(des1, des2, cv2.NORM_HAMMING))
        if H is not None:
            return H
    except Exception:
        pass
    return IDENTITY.copy()
