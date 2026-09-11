"""Local scorer. Mirrors the interface the README documents for evaluate.py.

  python evaluate.py --predictions preds.csv --ground_truth train.csv --data_dir data/train
"""
import argparse, csv, os, sys
import hscore

COLS = ("h11", "h12", "h13", "h21", "h22", "h23", "h31", "h32")


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def sizes_for(row, data_dir):
    """Image dims from explicit columns if present, else read off disk."""
    if all(k in row and row[k] for k in ("width_1", "height_1", "width_2", "height_2")):
        return ((int(row["width_1"]), int(row["height_1"])),
                (int(row["width_2"]), int(row["height_2"])))
    if not data_dir:
        sys.exit("--data_dir is required: ground truth has no width/height columns")
    return (hscore.image_size(os.path.join(data_dir, row["image_1"])),
            hscore.image_size(os.path.join(data_dir, row["image_2"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--ground_truth", required=True)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--per_pair", default=None, help="write per-pair errors here")
    a = ap.parse_args()

    preds = {r["pair_id"]: r for r in read_csv(a.predictions)}
    truth = read_csv(a.ground_truth)

    rows, errors = [], []
    missing = 0
    for t in truth:
        pid = t["pair_id"]
        p = preds.get(pid)
        if p is None:
            missing += 1
            e = hscore.PENALTY
        else:
            wh1, wh2 = sizes_for(t, a.data_dir)
            e = hscore.pair_error(hscore.to_matrix(p), hscore.to_matrix(t), wh1, wh2)
        errors.append(e)
        rows.append((pid, e))

    if not errors:
        sys.exit("no ground-truth rows scored")

    mean = sum(errors) / len(errors)
    penalised = sum(1 for e in errors if e >= hscore.PENALTY)
    rows.sort(key=lambda r: -r[1])

    print(f"pairs scored        {len(errors)}")
    print(f"missing predictions {missing}")
    print(f"penalised pairs     {penalised}")
    print(f"reprojection error  {mean:.6f}   (lower is better, 0 = perfect)")
    print(f"leaderboard_score   {hscore.leaderboard_score(mean):.5f}   (0-100, higher is better)")
    print("\nworst pairs:")
    for pid, e in rows[:10]:
        print(f"  {e:.6f}  {pid}")

    if a.per_pair:
        with open(a.per_pair, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pair_id", "error"])
            w.writerows(rows)


if __name__ == "__main__":
    main()
