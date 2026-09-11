"""Pre-flight a submission.csv against test.csv before spending a daily upload."""
import argparse, csv, sys
import numpy as np
import hscore

REQUIRED = ["pair_id", "h11", "h12", "h13", "h21", "h22", "h23", "h31", "h32"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()

    with open(a.submission, newline="") as f:
        r = csv.DictReader(f)
        header = r.fieldnames
        sub = list(r)
    with open(a.test, newline="") as f:
        want = [row["pair_id"] for row in csv.DictReader(f)]

    problems = []
    if header != REQUIRED:
        problems.append(f"header is {header}\n           expected {REQUIRED}")

    seen = [s["pair_id"] for s in sub]
    dupes = {p for p in seen if seen.count(p) > 1}
    if dupes:
        problems.append(f"{len(dupes)} duplicate pair_id(s), e.g. {sorted(dupes)[:5]}")

    missing = [p for p in want if p not in set(seen)]
    extra = [p for p in seen if p not in set(want)]
    if missing:
        problems.append(f"{len(missing)} missing pair_id(s), e.g. {missing[:5]}")
    if extra:
        problems.append(f"{len(extra)} unexpected pair_id(s), e.g. {extra[:5]}")

    bad_num, degenerate = [], []
    for s in sub:
        H = hscore.to_matrix(s)
        if H is None:
            bad_num.append(s["pair_id"])
        elif not hscore.is_usable(H):
            degenerate.append(s["pair_id"])
    if bad_num:
        problems.append(f"{len(bad_num)} non-numeric/NaN row(s), e.g. {bad_num[:5]}")
    if degenerate:
        problems.append(f"{len(degenerate)} singular H (would take the penalty), e.g. {degenerate[:5]}")

    print(f"rows {len(sub)}   expected {len(want)}")
    if problems:
        print("\nFAIL")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("OK - safe to upload")


if __name__ == "__main__":
    main()
