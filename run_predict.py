"""Run a predict() module over a pairs csv and write a submission-shaped csv.

  python run_predict.py --module baseline_sift.py --pairs data/test.csv \
      --data_dir data/test --out out/submission.csv
"""
import argparse, csv, importlib.util, os, sys, time
import numpy as np

COLS = ["pair_id", "h11", "h12", "h13", "h21", "h22", "h23", "h31", "h32"]


def load_module(path):
    spec = importlib.util.spec_from_file_location("user_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "predict"):
        sys.exit(f"{path} does not define predict(image_1_path, image_2_path)")
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    mod = load_module(a.module)
    with open(a.pairs, newline="") as f:
        pairs = list(csv.DictReader(f))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    t0 = time.perf_counter()
    fallbacks = 0
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for i, row in enumerate(pairs, 1):
            p1 = os.path.join(a.data_dir, row["image_1"])
            p2 = os.path.join(a.data_dir, row["image_2"])
            H = np.asarray(mod.predict(p1, p2), dtype=np.float64)
            if np.allclose(H, np.eye(3)):
                fallbacks += 1
            H = H / H[2, 2]
            w.writerow([row["pair_id"]] + [f"{v:.12g}" for v in
                                           (H[0, 0], H[0, 1], H[0, 2],
                                            H[1, 0], H[1, 1], H[1, 2],
                                            H[2, 0], H[2, 1])])
            if i % 20 == 0 or i == len(pairs):
                el = time.perf_counter() - t0
                print(f"  {i}/{len(pairs)}  {el:.1f}s  ({el/i:.2f}s/pair)", flush=True)

    print(f"\nwrote {a.out}")
    print(f"identity fallbacks: {fallbacks}/{len(pairs)}")


if __name__ == "__main__":
    main()
