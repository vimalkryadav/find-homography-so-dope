# Find Homography So Dope

Estimating the 3x3 homography mapping image 1 onto image 2 for each pair in a set
of 6-image scenes. Scored on mean reprojection error of five reference points
(four corners and the centre), reported on a 0-100 scale where higher is better.

The competition brief is in [COMPETITION.md](COMPETITION.md).

## Results

| version | approach | train | public |
|---|---|---|---|
| v5  | identity snap, chained composition, guided re-match | 98.48291 | **99.09285** |
| v6  | route consensus | 98.29446 | not submitted |
| v7  | consensus as veto | 98.43596 | not submitted |
| v8b | scene-graph bundle refinement | 98.72668 | 98.81967 |
| v9  | bundle gated on evidence strength | 98.52132 | not submitted |
| v10 | ECC photometric refinement | 98.86413 | not submitted |
| v11 | ECC accepted on a photometric test | 99.37509 | 98.23892 |
| v12 | ECC accepted on inlier agreement | 99.37509 | 99.00346 |
| v13 | inlier-residual tolerance at 1.7x | 99.28112 | 99.05830 |
| v14 | refine edges before composing | 99.28111 | not submitted |
| v15 | dense matching in the weak-inlier band | **99.42418** | pending |

## Approach

Half the scenes are illumination-only, where the true homography is exactly
identity. Classifying at the scene level rather than per pair separated all 28
training scenes with a wide margin and lets a scene's one bad pair be overridden
by its four good ones.

For the rest, transform magnitude grows with image index, so a 1 -> k pair too
wide to match directly can be recovered by composing shorter hops through
intermediate images. Inlier count decides between the direct edge and a chained
path.

Everything through v9 estimates the homography from sparse keypoints, so all of
it inherits SIFT's half-pixel localisation error no matter how the routes are
selected. That is why route selection, consensus and bundle adjustment all landed
within noise of each other. ECC refinement attacks the floor instead, aligning on
pixel intensities across the whole overlap, and is invariant to affine
illumination change.

## What the metric rewards

The scoring is unusually sensitive on wide-baseline pairs. `scene_010_1_5` maps a
corner to `w ~= -0.1`, past the horizon, and the metric divides by that `w`. A
homography matching ground truth to four decimal places still scores 0.179 on that
pair, because four of its five points are near-perfect and one is amplified about
a hundredfold. That single pair was 42% of all remaining training error before ECC.

Only one pair in the test set has that signature: `scene_023_1_6`, at min|w| =
0.1001 against `scene_010_1_5`'s 0.1085. It is reachable only through image 5,
and every version that tried to improve it moved it toward the same wrong basin.
v11 differed from v5 almost only there and scored 0.85 lower, which over 60 pairs
is about 0.102 of total error against a displacement of 0.1075 - arithmetic that
only works if the v5 answer is already close to correct.

The same sensitivity makes geometric gates misleading: judging a refinement by how
far the reference corners moved rejects exactly the pairs that need it most.
Measuring the residual on the matched inliers instead, where the features actually
are, separates a correct near-horizon correction from a photometric slide.

## Dense matching

`dense_matcher.py` runs LoFTR and is used only where the sparse matcher thins out,
between 30 and 200 inliers. Outside that band it does not earn its place: above it
the sparse fit is more precise, and below it the pair is hard enough that the dense
matcher fails too (scene_033 returns 8 matches and a useless homography). Raising
its input resolution does not help, since the model is trained near 840px and
degrades on either side.

Its answers are deliberately left unpolished. Photometric refinement pulls them
back to the sparse answer, which for exactly these pairs is the worse one:
scene_006_1_6 goes 0.0151 dense, 0.0287 after ECC, 0.0548 sparse.

The import is optional. Without torch and kornia installed, `solution_v15.py`
falls back to the sparse path and reproduces v13 exactly.

## Layout

```
solution.py          v5, the best public score
solution_v6..v13.py  the versions above, in order
out/                 submissions and per-version training predictions
evaluate.py          scoring, matching the competition metric
run_predict.py       runs a predict() module over a pairs csv
validate_submission.py
benchmark.py         runtime measurement
```

Image data is not tracked here; see the competition page for the dataset.

## Reproducing

```
python run_predict.py --module solution.py --pairs test.csv \
    --data_dir data/test --out out/submission.csv
python validate_submission.py --submission out/submission.csv --test test.csv
```

Self-check against the training ground truth:

```
python run_predict.py --module solution.py --pairs train.csv \
    --data_dir data/train --out out/train.csv
python evaluate.py --predictions out/train.csv --ground_truth train.csv \
    --data_dir data/train
```
