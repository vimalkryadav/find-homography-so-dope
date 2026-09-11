"""Reimplementation of the competition metric, per the dataset README.

The README references an evaluate.py that is not shipped in the data bundle,
so this is written from the documented definition:

  1. five normalized points in image 1: (0,0) (1,0) (1,1) (0,1) (0.5,0.5)
  2. to pixels via image 1's width/height
  3. warp with predicted H and with ground-truth H
  4. back to normalized coords via image 2's width/height
  5. euclidean distance per point, mean over the 5, mean over all pairs

Lower is better. The leaderboard shows 100 * max(0, 1 - error / 0.2).
"""
import numpy as np
from PIL import Image

CORNERS = np.array(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float64
)

# The README says "convert to pixel coordinates using Image 1's dimensions".
# Read literally that is u*W; the u*(W-1) reading differs by a sub-pixel amount.
# Flip this if a host clarification says otherwise.
PIXEL_SCALE_MINUS_ONE = False

# Penalty for a missing / non-numeric / degenerate H. The real scorer's constant
# is not published; anything well above the 0.2 display cutoff ranks the same.
PENALTY = 1.0

_dim_cache = {}


def image_size(path):
    """(width, height) without decoding pixel data."""
    if path not in _dim_cache:
        with Image.open(path) as im:
            _dim_cache[path] = im.size
    return _dim_cache[path]


def to_matrix(row):
    """8 free values -> 3x3 with h33 = 1. Returns None if unusable."""
    try:
        v = [float(row[k]) for k in
             ("h11", "h12", "h13", "h21", "h22", "h23", "h31", "h32")]
    except (KeyError, TypeError, ValueError):
        return None
    if not all(np.isfinite(v)):
        return None
    return np.array([[v[0], v[1], v[2]], [v[3], v[4], v[5]], [v[6], v[7], 1.0]])


def is_usable(H):
    if H is None or H.shape != (3, 3) or not np.all(np.isfinite(H)):
        return False
    return abs(float(np.linalg.det(H))) > 1e-12


def warp_normalized(H, wh1, wh2):
    """Warp the five reference points; return them normalized in image 2."""
    w1, h1 = wh1
    w2, h2 = wh2
    sx1, sy1 = (w1 - 1, h1 - 1) if PIXEL_SCALE_MINUS_ONE else (w1, h1)
    sx2, sy2 = (w2 - 1, h2 - 1) if PIXEL_SCALE_MINUS_ONE else (w2, h2)

    pts = np.column_stack([
        CORNERS[:, 0] * sx1,
        CORNERS[:, 1] * sy1,
        np.ones(len(CORNERS)),
    ])
    out = pts @ H.T
    w = out[:, 2]
    if np.any(np.abs(w) < 1e-12):
        return None
    return np.column_stack([out[:, 0] / w / sx2, out[:, 1] / w / sy2])


def pair_error(H_pred, H_true, wh1, wh2):
    """Mean reprojection error over the five points, or PENALTY."""
    if not is_usable(H_pred) or not is_usable(H_true):
        return PENALTY
    a = warp_normalized(H_pred, wh1, wh2)
    b = warp_normalized(H_true, wh1, wh2)
    if a is None or b is None:
        return PENALTY
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def leaderboard_score(error):
    return 100.0 * max(0.0, 1.0 - error / 0.2)
