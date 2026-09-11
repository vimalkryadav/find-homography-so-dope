"""Local stand-in for the official runtime benchmark (also not shipped).

  python benchmark.py --module baseline_sift.py --test data/test.csv --data_dir data/test

Times image loading, preprocessing, feature extraction, matching and homography
estimation, with a short warm-up excluded, using time.perf_counter().
"""
import argparse, csv, os, time
import numpy as np
from run_predict import load_module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--warmup", type=int, default=2)
    a = ap.parse_args()

    mod = load_module(a.module)
    with open(a.test, newline="") as f:
        pairs = list(csv.DictReader(f))
    paths = [(os.path.join(a.data_dir, r["image_1"]),
              os.path.join(a.data_dir, r["image_2"])) for r in pairs]

    for p1, p2 in paths[:a.warmup]:
        mod.predict(p1, p2)

    per = []
    for p1, p2 in paths:
        t = time.perf_counter()
        mod.predict(p1, p2)
        per.append(time.perf_counter() - t)

    per = np.array(per)
    print(f"pairs        {len(per)}")
    print(f"total        {per.sum():.2f}s")
    print(f"mean/pair    {per.mean():.3f}s")
    print(f"median/pair  {np.median(per):.3f}s")
    print(f"slowest pair {per.max():.3f}s")


if __name__ == "__main__":
    main()
