"""Scene-aware homography estimation.

Exposes predict(image_1_path, image_2_path) -> 3x3 ndarray, as the official
benchmark harness expects. Work is done per scene and cached, so the first call
for a scene is expensive and the remaining four are free.

Three things this exploits, each verified against train.csv:

1. Half the scenes are illumination-only, where the true H is exactly identity.
   Classifying at the SCENE level (median over its five estimates) separated all
   28 training scenes with a 41x margin, and lets a scene's one bad pair be
   overridden by its four good ones.
2. Transform magnitude grows with image index, so an unmatched 1->k can be
   recovered by composing shorter hops through intermediate images.
3. Inlier count is a reliable proxy for whether a match can be trusted, which is
   what decides between the direct edge and a chained path.
"""
import itertools
import os

import cv2
import numpy as np

NFEATURES = 8000
RATIO = 0.85
RANSAC_THRESH = 3.0
MAX_ITERS = 20000
CONFIDENCE = 0.9999
MIN_MATCHES = 4

# A direct edge at or above this many inliers is trusted as-is; below it we look
# for a chained path. Training separation was 79+ (good) vs 27- (broken).
MIN_DIRECT_INLIERS = 50

# Scene median distance-from-identity below this means an illumination scene.
# Training: illumination scenes topped out at 0.0048, viewpoint began at 0.196.
IDENTITY_SNAP = 0.01

# Guided re-match radii in pixels, applied in order to polish an estimate.
GUIDED_RADII = (24.0, 10.0, 4.0)

# Illumination sequences include day-vs-night pairs where the dark frames carry
# almost no SIFT response at native contrast. CLAHE equalises them locally
# without moving anything geometrically.
CLAHE_CLIP = 3.0
CLAHE_GRID = (8, 8)

# Photometric refinement. Every estimate so far is built from sparse keypoints,
# so all of them inherit SIFT's half-pixel localisation error no matter how the
# routes are chosen. ECC aligns on pixel intensities instead, over the whole
# overlap, and is invariant to affine illumination change. Coarse-to-fine, with
# the last pass at full resolution so no resampling error survives into the answer.
ECC_SCALES = (0.25, 0.5, 1.0)
ECC_ITERS = 200
ECC_EPS = 1e-7
ECC_GAUSS = 5

# A refinement that relocates the image is a diverged solve, not a refinement.
ECC_MAX_DRIFT = 0.05

CORNERS = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]])
IDENTITY = np.eye(3, dtype=np.float64)

_scene_cache = {}


def _rootsift(des):
    """L1-normalise then square-root: Hellinger distance under an L2 matcher."""
    if des is None:
        return None
    des = des / (des.sum(axis=1, keepdims=True) + 1e-7)
    return np.sqrt(des)


def _sanitise(H):
    if H is None:
        return None
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3) or not np.all(np.isfinite(H)) or abs(H[2, 2]) < 1e-12:
        return None
    H = H / H[2, 2]
    if not np.all(np.isfinite(H)) or abs(float(np.linalg.det(H))) < 1e-12:
        return None
    return H


AREA_RATIO_MIN = 1e-3
AREA_RATIO_MAX = 1e3


def _quad(H, wh):
    """Image rectangle mapped through H: (corners, area ratio) or None."""
    if H is None:
        return None
    w, h = wh
    if min(w, h) <= 0:
        return None
    c = np.array([[0.0, 0.0, 1.0], [w, 0.0, 1.0], [w, h, 1.0], [0.0, h, 1.0]])
    p = c @ H.T
    if not np.all(np.isfinite(p)) or np.any(np.abs(p[:, 2]) < 1e-9):
        return None
    q = p[:, :2] / p[:, 2:3]
    if not np.all(np.isfinite(q)):
        return None
    x, y = q[:, 0], q[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    return q, area / (w * h), np.sign(p[:, 2])


def _degenerate(H, wh):
    """Hard reject: only matrices that have visibly collapsed.

    Deliberately permissive about *extreme* transforms. A genuine ground-truth
    homography in this dataset can be non-convex with corners either side of the
    horizon (scene_010_1_5 is), so convexity and w-sign consistency are treated
    as quality hints in _clean(), never as grounds for throwing an estimate away.
    Degenerate RANSAC consensus sets collapse the rectangle to a sliver and are
    caught by the area ratio.
    """
    got = _quad(H, wh)
    if got is None:
        return True
    _, ratio, _ = got
    return not (AREA_RATIO_MIN < ratio < AREA_RATIO_MAX)


def _clean(H, wh):
    """Soft signal: the transform is convex and entirely in front of the camera."""
    got = _quad(H, wh)
    if got is None:
        return False
    q, ratio, signs = got
    if not np.all(signs == signs[0]):
        return False
    cross = []
    for t in range(4):
        a, b, c2 = q[t], q[(t + 1) % 4], q[(t + 2) % 4]
        u, v = b - a, c2 - b
        cross.append(u[0] * v[1] - u[1] * v[0])
    first = np.sign(cross[0])
    return bool(first != 0 and all(np.sign(v) == first for v in cross))


class _Scene:
    """Features, pairwise edges and per-target answers for one scene folder."""

    def __init__(self, folder):
        self.folder = folder
        self.sift = cv2.SIFT_create(nfeatures=NFEATURES)
        self.clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.idx = sorted(
            int(os.path.splitext(f)[0])
            for f in os.listdir(folder)
            if f.lower().endswith(".png") and os.path.splitext(f)[0].isdigit()
        )
        self._feat = {}
        self._gray = {}
        self.edges = {}          # (i, j) -> (H, inlier_count)
        self._tried = set()
        self.answers = {}
        self.unresolved = set()
        self._solve()

    # ---------- features and matching ----------

    def _features(self, i):
        if i not in self._feat:
            img = cv2.imread(os.path.join(self.folder, f"{i}.png"), cv2.IMREAD_GRAYSCALE)
            if img is None:
                self._feat[i] = ((), None, (0, 0))
            else:
                kp, des = self.sift.detectAndCompute(self.clahe.apply(img), None)
                h, w = img.shape[:2]
                self._feat[i] = (kp, _rootsift(des), (w, h))
        return self._feat[i]

    def size(self, i):
        return self._features(i)[2]

    def _grayscale(self, i):
        if i not in self._gray:
            img = cv2.imread(os.path.join(self.folder, f"{i}.png"), cv2.IMREAD_GRAYSCALE)
            if img is None:
                self._gray[i] = None
            else:
                eq = self.clahe.apply(img)
                self._gray[i] = eq.astype(np.float32) / 255.0
        return self._gray[i]

    def _ecc_refine(self, i, j, H):
        """Polish H by direct photometric alignment, coarse to fine."""
        a = self._grayscale(i)
        b = self._grayscale(j)
        if a is None or b is None or H is None:
            return H
        best = np.asarray(H, dtype=np.float64)
        for scale in ECC_SCALES:
            if scale == 1.0:
                ia, ib = a, b
            else:
                ia = cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                ib = cv2.resize(b, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if min(ia.shape[:2] + ib.shape[:2]) < 16:
                continue
            S = np.diag([scale, scale, 1.0])
            warp = S @ best @ np.linalg.inv(S)
            if not np.all(np.isfinite(warp)) or abs(warp[2, 2]) < 1e-12:
                return best
            warp = (warp / warp[2, 2]).astype(np.float32)
            crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_ITERS, ECC_EPS)
            try:
                _, warp = cv2.findTransformECC(ia, ib, warp, cv2.MOTION_HOMOGRAPHY,
                                               crit, None, ECC_GAUSS)
            except cv2.error:
                continue
            cand = _sanitise(np.linalg.inv(S) @ np.asarray(warp, dtype=np.float64) @ S)
            if cand is None or _degenerate(cand, self.size(i)):
                continue
            best = cand
        return best

    def _ratio_matches(self, des1, des2):
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return []
        out = {}
        for pair in self.matcher.knnMatch(des1, des2, k=2):
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < RATIO * n.distance:
                out[m.queryIdx] = m.trainIdx
        return list(out.items())

    def _fit(self, kp1, kp2, matches):
        if len(matches) < MIN_MATCHES:
            return None, 0
        src = np.float32([kp1[q].pt for q, _ in matches]).reshape(-1, 1, 2)
        dst = np.float32([kp2[t].pt for _, t in matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(
            src, dst, cv2.USAC_MAGSAC, ransacReprojThreshold=RANSAC_THRESH,
            maxIters=MAX_ITERS, confidence=CONFIDENCE,
        )
        H = _sanitise(H)
        if H is None:
            return None, 0
        return H, (0 if mask is None else int(mask.sum()))

    def _edge(self, i, j):
        """Match i -> j once, memoised, storing the inverse direction too."""
        if (i, j) in self.edges or (i, j) in self._tried:
            return self.edges.get((i, j))
        self._tried.add((i, j))
        self._tried.add((j, i))
        kp1, des1, _ = self._features(i)
        kp2, des2, _ = self._features(j)
        H, n = self._fit(kp1, kp2, self._ratio_matches(des1, des2))
        if H is not None and _degenerate(H, self.size(i)):
            H = None
        if H is not None:
            self.edges[(i, j)] = (H, n)
            inv = _sanitise(np.linalg.inv(H))
            if inv is not None:
                self.edges[(j, i)] = (inv, n)
        return self.edges.get((i, j))

    # ---------- geometry helpers ----------

    def _corner_shift(self, H, i, j):
        """Distance the five reference points move, in the metric's own units."""
        w1, h1 = self.size(i)
        w2, h2 = self.size(j)
        if min(w1, h1, w2, h2) <= 0:
            return np.inf
        pts = np.column_stack([CORNERS[:, 0] * w1, CORNERS[:, 1] * h1, np.ones(len(CORNERS))])
        a = pts @ H.T
        b = pts @ IDENTITY.T
        if np.any(np.abs(a[:, 2]) < 1e-12):
            return np.inf
        an = np.column_stack([a[:, 0] / a[:, 2] / w2, a[:, 1] / a[:, 2] / h2])
        bn = np.column_stack([b[:, 0] / b[:, 2] / w2, b[:, 1] / b[:, 2] / h2])
        return float(np.mean(np.linalg.norm(an - bn, axis=1)))

    def _corner_gap(self, Ha, Hb, target):
        """How far apart two estimates put the five reference points."""
        w1, h1 = self.size(1)
        w2, h2 = self.size(target)
        if min(w1, h1, w2, h2) <= 0:
            return np.inf
        pts = np.column_stack([CORNERS[:, 0] * w1, CORNERS[:, 1] * h1, np.ones(len(CORNERS))])
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            a, b = pts @ Ha.T, pts @ Hb.T
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            return np.inf
        if np.any(np.abs(a[:, 2]) < 1e-12) or np.any(np.abs(b[:, 2]) < 1e-12):
            return np.inf
        an = np.column_stack([a[:, 0] / a[:, 2] / w2, a[:, 1] / a[:, 2] / h2])
        bn = np.column_stack([b[:, 0] / b[:, 2] / w2, b[:, 1] / b[:, 2] / h2])
        if not np.all(np.isfinite(an)) or not np.all(np.isfinite(bn)):
            return np.inf
        return float(np.mean(np.linalg.norm(an - bn, axis=1)))

    def _guided_refine(self, i, j, H):
        """Re-match under the current H within shrinking radii, refitting each time."""
        kp1, des1, _ = self._features(i)
        kp2, des2, _ = self._features(j)
        if H is None or des1 is None or des2 is None or not len(kp1) or not len(kp2):
            return H
        p1 = np.array([k.pt for k in kp1], dtype=np.float64)
        p2 = np.array([k.pt for k in kp2], dtype=np.float64)
        best = H
        for radius in GUIDED_RADII:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                proj = np.column_stack([p1, np.ones(len(p1))]) @ best.T
            if not np.all(np.isfinite(proj)):
                break
            w = proj[:, 2]
            ok = np.abs(w) > 1e-12
            if ok.sum() < MIN_MATCHES:
                break
            proj = proj[ok, :2] / w[ok, None]
            src_idx = np.flatnonzero(ok)

            # nearest kp2 to each projected kp1, accepted inside the radius
            d = np.linalg.norm(proj[:, None, :] - p2[None, :, :], axis=2) \
                if len(p2) * len(proj) <= 4_000_000 else None
            if d is None:
                break
            nearest = np.argmin(d, axis=1)
            keep = d[np.arange(len(proj)), nearest] < radius
            if keep.sum() < MIN_MATCHES:
                break
            matches = list(zip(src_idx[keep], nearest[keep]))
            cand, n = self._fit(kp1, kp2, matches)
            if cand is None or n < MIN_MATCHES:
                break
            best = cand
        return best

    # ---------- solve ----------

    def _candidates(self, target):
        """Every route 1 -> target, as (H, bottleneck_inliers, path)."""
        out = []
        direct = self.edges.get((1, target))
        if direct is not None:
            out.append((direct[0], direct[1], (1, target)))
        others = [n for n in self.idx if n not in (1, target)]
        for r in range(1, min(3, len(others)) + 1):
            for mid in itertools.permutations(others, r):
                path = (1,) + mid + (target,)
                edges = [(path[t], path[t + 1]) for t in range(len(path) - 1)]
                for e in edges:
                    self._edge(*e)
                if any(e not in self.edges for e in edges):
                    continue
                H = IDENTITY.copy()
                for e in edges:
                    H = self.edges[e][0] @ H
                H = _sanitise(H)
                if H is None or _degenerate(H, self.size(1)):
                    continue
                out.append((H, min(self.edges[e][1] for e in edges), path))
        return out

    def _solve(self):
        targets = [k for k in self.idx if k != 1]

        # direct edges first: the common case needs nothing more
        direct = {k: self._edge(1, k) for k in targets}

        shifts = [self._corner_shift(e[0], 1, k)
                  for k, e in direct.items() if e is not None]
        if not shifts or float(np.median(shifts)) < IDENTITY_SNAP:
            for k in targets:
                self.answers[k] = IDENTITY.copy()
            return

        for k in targets:
            e = direct.get(k)
            if e is not None and e[1] >= MIN_DIRECT_INLIERS and not _degenerate(e[0], self.size(1)):
                best = e[0]
            else:
                # prefer a clean transform, then the strongest bottleneck, then
                # the shortest route, since each composition compounds error
                cands = self._candidates(k)
                if cands:
                    best = max(cands, key=lambda c: (_clean(c[0], self.size(1)),
                                                     c[1], -len(c[2])))[0]
                else:
                    best = e[0] if e is not None else None

            if best is None:
                self.unresolved.add(k)
                self.answers[k] = IDENTITY.copy()
                continue

            refined = self._guided_refine(1, k, best)
            if refined is not None and not _degenerate(refined, self.size(1)):
                best = refined

            polished = self._ecc_refine(1, k, best)
            if (polished is not None and not _degenerate(polished, self.size(1))
                    and self._corner_gap(best, polished, k) <= ECC_MAX_DRIFT):
                best = polished
            self.answers[k] = best


def predict(image_1_path: str, image_2_path: str) -> np.ndarray:
    try:
        folder = os.path.dirname(image_1_path)
        if folder not in _scene_cache:
            _scene_cache[folder] = _Scene(folder)
        scene = _scene_cache[folder]
        target = int(os.path.splitext(os.path.basename(image_2_path))[0])
        H = scene.answers.get(target)
        if H is not None:
            return np.asarray(H, dtype=np.float64)
    except Exception:
        pass
    return IDENTITY.copy()
