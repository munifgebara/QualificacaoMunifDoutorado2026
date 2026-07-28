# A2 — Anonymization ablation

Scripts for the paired anonymization experiment (Activity A2 of the research
plan). They exist to replace the single informal comparison reported in the
IWSSIP study (0.83 against 0.82, SVM, one run) with a paired, cross-validated
result carrying a statistical statement.

## Why the comparison can be paired

Anonymization is a deterministic per-character map (`encrypt_chars`) applied at
render time. It changes pixel intensities and leaves the geometry untouched:
same file, same dimensions, same layout, same label. `a2_generate.py` therefore
renders both versions **in a single pass**, so the two datasets are paired
instance by instance. Generating them in two independent runs would lose that
guarantee.

Pairing matters because paired tests are far more powerful than comparing two
independent runs, and because McNemar needs per-instance correspondence.

## The two questions

The claim the thesis defends is that anonymization costs *little*. That is an
**equivalence** claim, not a difference claim, and the two require different
tests:

| Question | Test | Reported as |
|---|---|---|
| Is there any detectable difference? | corrected paired t-test, McNemar | `p_difference` |
| Is the difference small enough to ignore? | TOST against a declared margin | `p_tost` |

A non-significant difference does **not** establish equivalence — with enough
noise you fail to reject simply for lack of power. The sanity checks in the
scripts include exactly that case. Declare the margin (`--margin`, default 0.02)
before running, not after seeing the result.

The paired t-test uses the Nadeau–Bengio correction, because folds of a repeated
cross-validation share training data and the naive t-test is anti-conservative.

## Usage

```bash
# 1. paired dataset (both renderings, one pass)
python a2_generate.py --src /path/to/100repos --out /path/to/dataset_a2 \
                      --min-per-class 200 --max-per-class 1000

# 2. paired evaluation with statistics
python a2_experiment.py --data /path/to/dataset_a2 \
                        --repeats 10 --folds 5 --margin 0.02

# 3. what the anonymization destroys, no model needed
python a2_entropy.py --src /path/to/100repos --per-class
```

Outputs: `a2_results.json`, `a2_results_table.tex` (drops straight into the
document), and the per-fold accuracies for both representations.

Add `--variable-size` to `a2_generate.py` to reproduce the variable-size variant
instead of 128×128.

## Cost

Measured at roughly 360–410 minimaps/s per core, with both renderings produced
in the same pass; generation is parallel over `--workers`. A corpus of 20 classes
capped at 1,000 samples runs in under a minute. Rendering all 685,906 images of
the large-scale corpus takes about half an hour on one core and a few minutes in
parallel.

For the evaluation, Random Forest over 59-dimensional LBP descriptors is cheap.
SVM with an RBF kernel is roughly quadratic in the number of samples, so on
corpora beyond ~20,000 minimaps either subsample or run `--classifiers rf` alone.

## Verified

Both scripts were run end to end on a small mixed corpus (15 classes, 3,546
paired minimaps). The statistical routines were checked against synthetic
differences of known size: a null difference is reported as equivalent, a 1-point
drop as significant but still equivalent within a 2-point margin, a 5-point drop
as neither, and a null difference under large noise as non-significant but *not*
equivalent — the failure mode the design is meant to expose.

## Scope

This is the two-level version (anonymization on/off) needed to retire the
"informal comparison". The graded ablation over several suppression levels, which
Activity A2 promises in full, reuses the same machinery: generate one dataset per
suppression level and compare each against the plain baseline.
