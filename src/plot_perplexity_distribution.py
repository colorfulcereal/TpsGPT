#!/usr/bin/env python3
"""
Plot a high-quality SVG histogram of perplexity scores.

Usage examples:
  python plot_perplexity_distribution.py data/perplexities.txt
  python plot_perplexity_distribution.py data/scores.csv --column perplexity --bins 60 --title "Perplexity Distribution"
  python plot_perplexity_distribution.py data/scores.tsv --column ppl --outfile my_plot.svg

Input file formats supported:
  - Plain text with one numeric score per line
  - CSV/TSV with a named column (use --column)
  - Space-delimited with a single numeric column

The script automatically:
  - Cleans NaNs and non-finite values
  - Prints summary stats to stdout
  - Saves a vector (SVG) figure suitable for publication

Note:
  - Uses matplotlib only (no seaborn), one chart, no explicit color settings.
"""

import argparse
import math
import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_scores(path: str, column: Optional[str]) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()

    # Try the most robust path first if a column is specified
    if column:
        try:
            df = pd.read_csv(path)
        except Exception:
            # Try tab
            df = pd.read_csv(path, sep='\t')
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found. Available columns: {list(df.columns)}")
        series = pd.to_numeric(df[column], errors='coerce')
        return series.to_numpy()

    # If no column provided, try reading as a single-column file
    # Attempt CSV with no header
    try:
        df = pd.read_csv(path, header=None)
        if df.shape[1] == 1:
            return pd.to_numeric(df.iloc[:, 0], errors='coerce').to_numpy()
    except Exception:
        pass

    # Try tab-delimited with no header
    try:
        df = pd.read_csv(path, sep='\t', header=None)
        if df.shape[1] == 1:
            return pd.to_numeric(df.iloc[:, 0], errors='coerce').to_numpy()
    except Exception:
        pass

    # As a last resort, try python engine with sep=None (guessing)
    try:
        df = pd.read_csv(path, sep=None, engine='python')
        if df.shape[1] == 1:
            return pd.to_numeric(df.iloc[:, 0], errors='coerce').to_numpy()
        else:
            raise ValueError(
                "Multiple columns detected. Please specify which with --column.")
    except Exception as e:
        raise RuntimeError(f"Could not parse input file. Specify --column if it's a CSV/TSV. Original error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Plot SVG histogram of perplexity scores.")
    parser.add_argument('infile', help='Path to perplexity file (txt/csv/tsv).')
    parser.add_argument('--column', help='Column name if infile has multiple columns.')
    parser.add_argument('--bins', type=int, default=50, help='Number of histogram bins (default: 50).')
    parser.add_argument('--title', type=str, default='Perplexity Score Distribution', help='Figure title.')
    parser.add_argument('--outfile', type=str, default='perplexity_distribution.svg', help='Output SVG filename.')
    parser.add_argument('--xlim', type=float, nargs=2, metavar=('XMIN', 'XMAX'), help='Optional x-axis limits.')
    parser.add_argument('--density', action='store_true', help='Normalize histogram to probability density.')

    args = parser.parse_args()

    scores = load_scores(args.infile, args.column)
    # Clean values
    scores = pd.to_numeric(pd.Series(scores), errors='coerce')
    scores = scores.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()

    if scores.size == 0:
        raise ValueError('No valid numeric scores found after cleaning.')

    # Basic stats
    q1 = float(np.percentile(scores, 25))
    q2 = float(np.percentile(scores, 50))
    q3 = float(np.percentile(scores, 75))
    mean = float(np.mean(scores))
    std = float(np.std(scores, ddof=1)) if scores.size > 1 else float('nan')
    print(f"Loaded {scores.size} scores from: {args.infile}")
    print(f"Mean={mean:.4f}  Std={std:.4f}  Median={q2:.4f}  IQR=({q1:.4f}, {q3:.4f})")
    print(f"Min={scores.min():.4f}  Max={scores.max():.4f}")

    # Create figure
    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()

    # Histogram
    ax.hist(scores, bins=args.bins, edgecolor='black', linewidth=0.5, density=args.density)

    # Labels and grid
    ax.set_title(args.title)
    ax.set_xlabel('Perplexity')
    ax.set_ylabel('Density' if args.density else 'Count')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)

    if args.xlim:
        ax.set_xlim(args.xlim)

    # Tidy layout and save as SVG
    plt.tight_layout()
    fig.savefig(args.outfile, format='svg')
    print(f"Saved SVG to: {args.outfile}")


if __name__ == '__main__':
    main()