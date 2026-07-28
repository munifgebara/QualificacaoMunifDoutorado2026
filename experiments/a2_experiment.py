#!/usr/bin/env python3
"""
a2_experiment.py -- paired anonymization ablation with statistical testing (A2).

Consumes the paired dataset produced by a2_generate.py and answers two distinct
questions that are frequently confused:

  1. Is there a detectable difference between the anonymized and the plain
     representation?  ->  corrected paired t-test and McNemar.
  2. Is the difference small enough to be irrelevant?  ->  TOST equivalence test
     against a margin declared in advance.

Question 2 is the one the thesis actually needs. A non-significant result in
question 1 does NOT establish equivalence: it can simply mean the experiment
lacked power. Reporting only question 1 is the mistake this script exists to
avoid.

Design notes:
  * The two representations are evaluated on exactly the same cross-validation
    splits, so every comparison is paired.
  * Repeated stratified k-fold is used instead of a single holdout, so the
    difference comes with a variance estimate.
  * The paired t-test uses the Nadeau-Bengio correction, because folds of a
    repeated CV overlap in their training data and the naive t-test is
    anti-conservative.

Usage:
    python a2_experiment.py --data /path/to/dataset_a2
    python a2_experiment.py --data ... --margin 0.02 --repeats 10 --folds 5
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
from PIL import Image
from scipy import stats
from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC

# Descriptor configuration, matching lbp_example.py from munifgebara/codeminimap.
LBP_P, LBP_R, LBP_METHOD, LBP_BINS = 8, 2, 'nri_uniform', 59


def describe(path):
    """Returns the normalized LBP histogram, or None if the image cannot be
    read. A truncated file must invalidate the instance in BOTH variants, never
    only in one, or the pairing silently breaks."""
    try:
        img = np.array(Image.open(path).convert('L'))
    except Exception:
        return None
    lbp = local_binary_pattern(img, LBP_P, LBP_R, method=LBP_METHOD)
    hist, _ = np.histogram(lbp.ravel(), bins=LBP_BINS, range=(0, LBP_BINS))
    hist = hist.astype('float')
    return hist / (hist.sum() + 1e-7)


def load_variant(root, variant, order, cache_dir=None):
    """Loads descriptors in a fixed (label, image) order so that row i of the
    anon matrix and row i of the plain matrix are the same source file.

    Descriptors are cached on disk because reading tens of thousands of small
    PNGs dominates the runtime on network or virtualised mounts, and because it
    lets several evaluations reuse a single extraction pass.
    """
    cache = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache = os.path.join(cache_dir, f'lbp_{variant}_{len(order)}.npy')
        if os.path.exists(cache):
            X = np.load(cache)
            if X.shape == (len(order), LBP_BINS):
                print(f'  {variant}: reusing cached descriptors')
                return X

    X = np.empty((len(order), LBP_BINS), dtype=np.float64)
    for i, (label, name) in enumerate(order):
        X[i] = describe(os.path.join(root, variant, label, name))
        if (i + 1) % 2000 == 0:
            print(f'  {variant}: {i + 1}/{len(order)}', flush=True)
    if cache:
        np.save(cache, X)
    return X


def build_order(root):
    anon_root = os.path.join(root, 'anon')
    plain_root = os.path.join(root, 'plain')
    order = []
    for label in sorted(os.listdir(anon_root)):
        a = set(os.listdir(os.path.join(anon_root, label)))
        p = set(os.listdir(os.path.join(plain_root, label)))
        common = a & p
        if len(common) != len(a) or len(common) != len(p):
            print(f'  warning: {label} has {len(a)} anon and {len(p)} plain '
                  f'images; using the {len(common)} paired ones')
        order += [(label, n) for n in sorted(common)]
    return order


def make_classifier(name, seed):
    if name == 'rf':
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    if name == 'svm':
        return SVC(C=1000, gamma=0.01, kernel='rbf', random_state=seed)
    raise ValueError(name)


def corrected_paired_t(diffs, n_train, n_test):
    """Nadeau-Bengio corrected resampled t-test.

    Returns (mean, corrected standard error, t statistic, two-sided p, df).
    """
    d = np.asarray(diffs, dtype=float)
    J = len(d)
    mean = d.mean()
    var = d.var(ddof=1)
    correction = 1.0 / J + n_test / n_train
    se = np.sqrt(var * correction) if var > 0 else 0.0
    df = J - 1
    if se == 0:
        return mean, 0.0, np.inf if mean else 0.0, 0.0 if mean else 1.0, df
    t = mean / se
    p = 2 * stats.t.sf(abs(t), df)
    return mean, se, t, p, df


def tost(mean, se, df, margin):
    """Two one-sided tests for equivalence within +/- margin.

    Returns (p_value, is_equivalent) where p_value is the larger of the two
    one-sided p-values.
    """
    if se == 0:
        return (0.0, True) if abs(mean) < margin else (1.0, False)
    t_lower = (mean + margin) / se      # H0: mean <= -margin
    t_upper = (mean - margin) / se      # H0: mean >= +margin
    p_lower = stats.t.sf(t_lower, df)
    p_upper = stats.t.cdf(t_upper, df)
    p = max(p_lower, p_upper)
    return p, p < 0.05


def mcnemar(correct_a, correct_b):
    """Exact/corrected McNemar on paired per-instance correctness.

    correct_a, correct_b: boolean arrays over the same instances.
    Returns (b, c, p_value) where b favours A and c favours B.
    """
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c
    if n == 0:
        return b, c, 1.0
    if n < 25:
        p = min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5))
    else:
        chi2 = (abs(b - c) - 1) ** 2 / n
        p = stats.chi2.sf(chi2, 1)
    return b, c, p


def run(X_plain, X_anon, y, clf_name, repeats, folds, seed, margin):
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    acc_plain, acc_anon = [], []
    oof_plain = np.zeros((repeats, len(y)), dtype=bool)
    oof_anon = np.zeros((repeats, len(y)), dtype=bool)
    n_train = n_test = 0

    for j, (tr, te) in enumerate(cv.split(X_plain, y)):
        rep = j // folds
        n_train, n_test = len(tr), len(te)
        for X, accs, oof in ((X_plain, acc_plain, oof_plain),
                             (X_anon, acc_anon, oof_anon)):
            clf = make_classifier(clf_name, seed + j)
            clf.fit(X[tr], y[tr])
            pred = clf.predict(X[te])
            accs.append(float(np.mean(pred == y[te])))
            oof[rep, te] = (pred == y[te])
        if (j + 1) % folds == 0:
            print(f'    repeat {rep + 1}/{repeats} done '
                  f'(plain {np.mean(acc_plain[-folds:]):.4f} | '
                  f'anon {np.mean(acc_anon[-folds:]):.4f})')

    diffs = np.array(acc_plain) - np.array(acc_anon)   # cost of anonymizing
    mean, se, t, p_diff, df = corrected_paired_t(diffs, n_train, n_test)
    p_tost, equivalent = tost(mean, se, df, margin)
    ci = (mean - stats.t.ppf(0.975, df) * se, mean + stats.t.ppf(0.975, df) * se)

    mcn = [mcnemar(oof_plain[r], oof_anon[r]) for r in range(repeats)]

    return {
        'classifier': clf_name,
        'accuracy_plain_mean': float(np.mean(acc_plain)),
        'accuracy_plain_std': float(np.std(acc_plain, ddof=1)),
        'accuracy_anon_mean': float(np.mean(acc_anon)),
        'accuracy_anon_std': float(np.std(acc_anon, ddof=1)),
        'mean_difference': float(mean),
        'ci95': [float(ci[0]), float(ci[1])],
        'corrected_t': float(t),
        'p_difference': float(p_diff),
        'equivalence_margin': margin,
        'p_tost': float(p_tost),
        'equivalent_within_margin': bool(equivalent),
        'mcnemar_per_repeat': [{'b_plain_only': b, 'c_anon_only': c, 'p': float(p)}
                               for b, c, p in mcn],
        'mcnemar_median_p': float(np.median([p for _, _, p in mcn])),
        'n_folds': len(diffs),
    }


def latex_table(results, n_classes, n_samples):
    lines = [
        r'\begin{table}[ht]',
        r'\centering',
        r'\caption{Anonymization ablation: paired comparison over '
        f'{n_classes} classes and {n_samples:,} minimaps. '
        r'The difference is the accuracy lost by anonymizing.}',
        r'\label{tab:a2-anonymization}',
        r'\begin{tabular}{lccccc}',
        r'\toprule',
        r'\textbf{Classifier} & \textbf{Plain} & \textbf{Anonymized} & '
        r'\textbf{Difference} & \textbf{95\% CI} & \textbf{TOST} \\',
        r'\midrule',
    ]
    for r in results:
        verdict = 'equivalent' if r['equivalent_within_margin'] else 'inconclusive'
        lines.append(
            f"{r['classifier'].upper()} & "
            f"{r['accuracy_plain_mean']:.4f} & "
            f"{r['accuracy_anon_mean']:.4f} & "
            f"{r['mean_difference']:+.4f} & "
            f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] & "
            f"{verdict} \\\\"
        )
    lines += [r'\bottomrule', r'\end{tabular}', r'\fautor', r'\end{table}']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='output directory of a2_generate.py')
    ap.add_argument('--classifiers', default='rf,svm')
    ap.add_argument('--repeats', type=int, default=10)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--margin', type=float, default=0.02,
                    help='equivalence margin in accuracy points (default 0.02)')
    ap.add_argument('--seed', type=int, default=18081975)
    ap.add_argument('--out', default=None, help='results JSON (default <data>/a2_results.json)')
    args = ap.parse_args()

    print('pairing images ...')
    order = build_order(args.data)
    if not order:
        sys.exit('no paired images found')
    y = np.array([label for label, _ in order])
    counts = Counter(y)
    print(f'  {len(counts)} classes, {len(order)} paired minimaps')
    for label, n in sorted(counts.items()):
        print(f'    {label:22s} {n:6d}')

    print('extracting LBP descriptors ...')
    X_plain = load_variant(args.data, 'plain', order)
    X_anon = load_variant(args.data, 'anon', order)

    results = []
    for clf_name in args.classifiers.split(','):
        clf_name = clf_name.strip()
        print(f'\n{clf_name.upper()}: {args.repeats}x{args.folds} repeated stratified CV')
        results.append(run(X_plain, X_anon, y, clf_name,
                           args.repeats, args.folds, args.seed, args.margin))

    out = args.out or os.path.join(args.data, 'a2_results.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'n_classes': len(counts), 'n_samples': len(order),
                   'margin': args.margin, 'results': results}, f, indent=2)

    print('\n' + '=' * 72)
    for r in results:
        print(f"\n{r['classifier'].upper()}")
        print(f"  plain      : {r['accuracy_plain_mean']:.4f} +/- {r['accuracy_plain_std']:.4f}")
        print(f"  anonymized : {r['accuracy_anon_mean']:.4f} +/- {r['accuracy_anon_std']:.4f}")
        print(f"  difference : {r['mean_difference']:+.4f}  "
              f"95% CI [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]")
        print(f"  difference test : p = {r['p_difference']:.4g} "
              f"({'significant' if r['p_difference'] < 0.05 else 'not significant'})")
        print(f"  equivalence within +/-{args.margin:.3f} : p = {r['p_tost']:.4g} "
              f"({'EQUIVALENT' if r['equivalent_within_margin'] else 'INCONCLUSIVE'})")
        print(f"  McNemar median p : {r['mcnemar_median_p']:.4g}")

    tex = os.path.splitext(out)[0] + '_table.tex'
    with open(tex, 'w', encoding='utf-8') as f:
        f.write(latex_table(results, len(counts), len(order)))
    print(f'\nresults: {out}\nLaTeX table: {tex}')


if __name__ == '__main__':
    main()
