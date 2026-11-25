#!/usr/bin/env python3
"""
Plot CDF of pLDDT scores (scaled to %), with marker at pLDDT ≥ 70%.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_scores(path):
    # Load single-column numeric file
    df = pd.read_csv(path, header=None)
    scores = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    scores = scores.dropna().to_numpy()

    # Scale from [0,1] to [0,100]
    return scores * 100


def main():
    parser = argparse.ArgumentParser(description="Plot CDF of pLDDT scores in %.")
    parser.add_argument("infile", help="File with pLDDT scores (0–1 scale).")
    parser.add_argument("--outfile", default="plddt_cdf.svg", help="Output SVG filename.")
    args = parser.parse_args()

    scores = load_scores(args.infile)

    # Sort values for CDF
    scores_sorted = np.sort(scores)
    cdf = np.arange(1, len(scores_sorted) + 1) / len(scores_sorted)

    # Fraction with pLDDT >= 70
    frac_high = np.mean(scores >= 70)

    # Coordinates for marker
    y_at_70 = np.searchsorted(scores_sorted, 70, side="left") / len(scores_sorted)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(scores_sorted, cdf, linewidth=2, label="CDF")

    # Marker
    ax.plot([70], [y_at_70], "ro", label=f"pLDDT ≥ 70 ({frac_high:.1%})")

    # Labels and formatting
    ax.set_title("CDF of pLDDT Scores")
    ax.set_xlabel("pLDDT")
    ax.set_ylabel("Cumulative Probability")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend()

    plt.tight_layout()
    fig.savefig(args.outfile, format="svg")
    print(f"Saved CDF plot to: {args.outfile}")
    print(f"Fraction of sequences with pLDDT ≥ 70%: {frac_high:.2%}")


if __name__ == "__main__":
    main()