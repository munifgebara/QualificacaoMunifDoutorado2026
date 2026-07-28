#!/usr/bin/env python3
"""
Regenerates figuras/lbp-histograms.png (Figure 6 of the qualification document).

The previous version of the figure had overlapping axis labels and, more
importantly, hid the fact that two LBP codes dominate every histogram. With
P = 8 and R = 2 under the nri_uniform mapping, code 57 corresponds to a flat
neighbourhood and code 58 is the catch-all for non-uniform patterns; together
they account for roughly 95% of the mass of every minimap, because a minimap is
mostly uniform background. All discriminative structure lives in the remaining
codes.

This version therefore shows two panels: the complete histogram on a
logarithmic scale, and a linear zoom over codes 0-56, where the classes
actually separate.

Descriptor configuration matches lbp_example.py from munifgebara/codeminimap:
    P = 8 sampling points, R = 2 radius, method = "nri_uniform"  -> 59 bins,
    histogram normalized to sum to one.

Usage:
    python gen_lbp_histograms.py --dataset <path to all_encrypted_fixed_size> \
                                 --out lbp-histograms.png
"""
import argparse
import os
import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P, R, METHOD, N_BINS = 8, 2, "nri_uniform", 59
FLAT_CODE, NONUNIFORM_CODE = 57, 58

CLASSES = [
    ("javaentity", "Java entity"),
    ("javaimplementation", "Java implementation"),
    ("javajsp", "JSP page"),
    ("js", "JavaScript"),
    ("json", "JSON"),
    ("svg", "SVG"),
]

COLORS = ["#1b4965", "#e07a5f", "#3d5a4c", "#9a6fb0", "#c9a227", "#8c8c8c"]


def describe(path):
    img = np.array(Image.open(path).convert("L"))
    lbp = local_binary_pattern(img, P, R, method=METHOD)
    hist, _ = np.histogram(lbp.ravel(), bins=N_BINS, range=(0, N_BINS))
    hist = hist.astype("float")
    return hist / (hist.sum() + 1e-7)


def mean_histogram(folder, cap):
    files = sorted(f for f in os.listdir(folder)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))[:cap]
    return np.mean([describe(os.path.join(folder, f)) for f in files], axis=0), len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="lbp-histograms.png")
    ap.add_argument("--cap", type=int, default=1000,
                    help="samples per class, matching the study's class cap")
    args = ap.parse_args()

    means = {}
    for folder, label in CLASSES:
        m, n = mean_histogram(os.path.join(args.dataset, folder), args.cap)
        means[label] = m
        print(f"{label:22s} n={n:4d}  flat={m[FLAT_CODE]:.3f}  "
              f"non-uniform={m[NONUNIFORM_CODE]:.3f}  rest={m[:FLAT_CODE].sum():.3f}")

    bins = np.arange(N_BINS)
    fig, (ax_full, ax_zoom) = plt.subplots(2, 1, figsize=(9, 7.5),
                                           constrained_layout=True)

    for (folder, label), color in zip(CLASSES, COLORS):
        ax_full.step(bins, means[label], where="mid", color=color,
                     linewidth=1.4, label=label)
    ax_full.set_yscale("log")
    ax_full.set_xlim(-1, N_BINS)
    ax_full.set_ylabel("Relative frequency (log)", fontsize=11)
    ax_full.set_title("Complete descriptor: codes 57 and 58 dominate every class",
                      fontsize=11, pad=6)
    ax_full.axvspan(FLAT_CODE - 0.5, NONUNIFORM_CODE + 0.5, color="#d94f4f", alpha=0.10)
    ax_full.annotate("flat and non-uniform\nneighbourhoods",
                     xy=(FLAT_CODE, 0.5), xytext=(41, 0.10), fontsize=9,
                     ha="center", color="#9c2b2b",
                     arrowprops=dict(arrowstyle="->", color="#9c2b2b", lw=0.9))
    ax_full.set_ylim(1e-6, 2.0)
    ax_full.grid(alpha=0.25, linewidth=0.5)
    ax_full.set_axisbelow(True)

    zoom_max = FLAT_CODE
    for (folder, label), color in zip(CLASSES, COLORS):
        ax_zoom.step(bins[:zoom_max], means[label][:zoom_max], where="mid",
                     color=color, linewidth=1.5, label=label)
    ax_zoom.set_xlim(-1, zoom_max)
    ax_zoom.set_xlabel(f"LBP code (nri_uniform, P = {P}, R = {R})", fontsize=11)
    ax_zoom.set_ylabel("Relative frequency", fontsize=11)
    ax_zoom.set_title("Discriminative region: codes 0-56, linear scale",
                      fontsize=11, pad=6)
    ax_zoom.grid(alpha=0.25, linewidth=0.5)
    ax_zoom.set_axisbelow(True)
    ax_zoom.legend(fontsize=9, ncol=2, frameon=False, loc="upper right")

    for ax in (ax_full, ax_zoom):
        ax.tick_params(labelsize=9)

    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
