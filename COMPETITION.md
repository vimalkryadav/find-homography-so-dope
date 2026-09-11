# Find Homography So Dope

Predict the homography that maps Image 1 onto Image 2 for each given pair.

## Objective

Given a pair of images, predict the 3x3 homography matrix that maps
**Image 1 -> Image 2**. Any approach is allowed: classical (SIFT/ORB +
matching + RANSAC), learned (CNN/ResNet), hybrid, or pretrained/generative
models. Two things are scored:

1. **Homography prediction quality** (main leaderboard metric)
2. **Inference runtime** (measured separately, not self-reported)

## Dataset

A curated subset of image sequences, split so that **entire sequences** go to
either train or test (no leakage between near-duplicate images from the same
scene).

```
data/
    train/<scene_id>/*.png
    test/<scene_id>/*.png
```

Each sequence has 6 images; pairs are formed as (1, k) for k = 2..6, with
image 1 as the reference (Image 1 in each pair). Scene ids are anonymized
sequential labels — don't read anything into the numbering.

## Input / Output

**train.csv**
```
pair_id,image_1,image_2,h11,h12,h13,h21,h22,h23,h31,h32
```

**test.csv**
```
pair_id,image_1,image_2
```

**sample_submission.csv** / your submission:
```
pair_id,h11,h12,h13,h21,h22,h23,h31,h32
```

`h33` is assumed to be `1` (standard homography normalization).

## Homography definition

For a point `(x1, y1)` in Image 1 (pixel coordinates), the homography H maps
it to a point `(x2, y2)` in Image 2 via:

```
[x2']   [h11 h12 h13] [x1]
[y2'] = [h21 h22 h23] [y1]
[w2']   [h31 h32   1] [ 1]

x2 = x2' / w2'
y2 = y2' / w2'
```

## Evaluation metric

We do **not** compare matrices element-by-element (H is only defined up to
scale, and small element errors can mean very different amounts of visual
misalignment). Instead we use **geometric reprojection error**:

For every test pair:
1. Take 5 normalized points in Image 1: `(0,0), (1,0), (1,1), (0,1), (0.5,0.5)`.
2. Convert to pixel coordinates using Image 1's dimensions.
3. Warp with the predicted H and with the hidden ground-truth H.
4. Convert both warped points to normalized coordinates in Image 2.
5. Compute the Euclidean distance between the predicted and ground-truth
   normalized points, for each of the 5 points, and average.

The competition score is the **mean reprojection error across all test
pairs** (lower is better).

**Note:** the Kaggle leaderboard itself displays this as a *Homography Match
Score* on a 0-100 scale (`100 * max(0, 1 - error / 0.2)`), so **higher is
better** there — same ranking, just inverted for display. `evaluate.py`
(below) reports both the raw, lower-is-better error and this same 0-100
`leaderboard_score`, so your local self-check matches what you'll see on
Kaggle.

Run it yourself with:
```
python evaluate.py --predictions predictions.csv --ground_truth ground_truth.csv
```

You can self-check locally on your own training pairs (which include the ground-truth
H) before submitting, by pointing `--ground_truth` at `train.csv` and adding `--data_dir`
so image dimensions can be read from disk (train.csv has no width/height columns):
```
python evaluate.py --predictions train_predictions.csv --ground_truth train.csv --data_dir data/train
```

## Submission format

Submit a single `submission.csv`:
```
pair_id,h11,h12,h13,h21,h22,h23,h31,h32
```

Validate before submitting:
```
python validate_submission.py --submission submission.csv --test test.csv
```

**Naming convention:** Kaggle identifies a submission by your logged-in account, not by
the uploaded file's name — you can upload a file called `submission.csv` regardless of
what it was named on your machine. For grading, you will also be asked to send a copy of
your exact submission named `<roll_number>_submission.csv` (e.g. `CS21B001_submission.csv`)
through whatever channel your instructor specifies.

## Runtime measurement

Runtime is **not** self-reported — it's benchmarked on the same machine for every student
after the deadline, using:
```
python benchmark.py --module your_script.py --test test.csv --data_dir data/test
```

Your script must expose:
```python
def predict(image_1_path: str, image_2_path: str) -> np.ndarray:
    ...  # returns a 3x3 homography, Image 1 -> Image 2
```

Runtime includes image loading, preprocessing, feature extraction/model
inference, matching, and homography estimation. It does **not** include
model training. You can and should run this yourself beforehand to check
your own runtime — it uses `time.perf_counter()` with a short warm-up before
timing, the same way it'll be measured officially.

## Baselines

- `baseline_sift.py` — SIFT -> BFMatcher -> Lowe ratio test -> RANSAC -> Homography
- `baseline_orb.py` — ORB -> BFMatcher (Hamming) -> Lowe ratio test -> RANSAC -> Homography (optional second baseline)

## Allowed methods

Anything: SIFT, ORB, BFMatcher, FLANN, CNN/ResNet, hybrid approaches,
pretrained models, generative models. No restrictions beyond producing a
valid `submission.csv`.

## Grading

The leaderboard ranks by homography reprojection error (lower is better; shown inverted
as a Homography Match Score on Kaggle itself — see the note above). Runtime is added
afterward as a separate column, not folded into a combined score.
