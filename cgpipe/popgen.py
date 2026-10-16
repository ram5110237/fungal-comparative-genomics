"""Population-genetic summaries on a samples x sites dosage matrix."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .io import Variant


def genotype_matrix(variants: Sequence[Variant]) -> np.ndarray:
    """Sites x samples matrix of alternate-allele frequencies within each sample
    (dosage / ploidy): 0/1 for haploids, 0/0.5/1 for diploids, NaN missing."""
    if not variants:
        return np.empty((0, 0))
    m = np.vstack([v.dosage / np.where(v.ploidy > 0, v.ploidy, np.nan) for v in variants])
    return m


def group_frequencies(G: np.ndarray, groups: np.ndarray, group_names: Sequence[str]):
    """Per-site allele frequency and non-missing sample count for each group."""
    freqs, ns = {}, {}
    for g in group_names:
        sub = G[:, groups == g]
        ns[g] = np.sum(~np.isnan(sub), axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            freqs[g] = np.nansum(sub, axis=1) / ns[g]
    return freqs, ns


def hudson_fst(p1, p2, n1, n2):
    """Hudson's Fst (Bhatia et al. 2013) returned as per-site numerator and
    denominator. Average genome-wide or in windows as a *ratio of averages*,
    never as the average of per-site ratios."""
    p1, p2, n1, n2 = map(np.asarray, (p1, p2, n1, n2))
    with np.errstate(invalid="ignore", divide="ignore"):
        num = (p1 - p2) ** 2 - p1 * (1 - p1) / (n1 - 1) - p2 * (1 - p2) / (n2 - 1)
        den = p1 * (1 - p2) + p2 * (1 - p1)
    bad = (n1 < 2) | (n2 < 2) | np.isnan(p1) | np.isnan(p2)
    num = np.where(bad, np.nan, num); den = np.where(bad, np.nan, den)
    return num, den


def ratio_of_averages(num, den) -> float:
    ok = ~np.isnan(num) & ~np.isnan(den)
    return float(np.sum(num[ok]) / np.sum(den[ok])) if ok.any() and np.sum(den[ok]) > 0 else np.nan


def site_pi(p, n):
    """Unbiased per-site heterozygosity / nucleotide diversity, 2p(1-p) n/(n-1)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 1, 2 * p * (1 - p) * n / (n - 1), np.nan)


def windowed_stats(sites: pd.DataFrame, chrom_lengths: Dict[str, int], window: int,
                   group_names: Sequence[str]) -> pd.DataFrame:
    """Non-overlapping windows: SNP count, pi per bp for each group, Hudson Fst.

    pi is divided by the window length (all sites assumed callable); for real
    data supply a callable-sites mask and divide by callable length instead.
    """
    rows = []
    for chrom, length in chrom_lengths.items():
        sub = sites[sites.chrom == chrom]
        for start in range(0, length, window):
            end = min(start + window, length)
            w = sub[(sub.pos > start) & (sub.pos <= end)]
            row = {"chrom": chrom, "start": start + 1, "end": end, "n_snps": len(w)}
            for g in group_names:
                row[f"pi_{g}"] = float(np.nansum(w[f"pi_{g}"])) / (end - start) if len(w) else 0.0
            row["fst_hudson"] = ratio_of_averages(w.fst_num.values, w.fst_den.values) if len(w) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def pairwise_distance(G: np.ndarray) -> np.ndarray:
    """Proportion of differing genotypes between samples over sites called in both
    (allele-sharing distance; equals the p-distance for haploids)."""
    n = G.shape[1]
    D = np.zeros((n, n))
    ok = ~np.isnan(G)
    for i in range(n):
        for j in range(i + 1, n):
            both = ok[:, i] & ok[:, j]
            d = np.mean(np.abs(G[both, i] - G[both, j])) if both.any() else np.nan
            D[i, j] = D[j, i] = d
    return D


def pca(G: np.ndarray, n_components: int = 5):
    """PCA of samples on mean-imputed, centred, standardised genotypes
    (Patterson et al. 2006 style scaling). Returns scores and variance explained."""
    X = G.T.copy()                                   # samples x sites
    mu = np.nanmean(X, axis=0)
    idx = np.where(np.isnan(X))
    X[idx] = np.take(mu, idx[1])
    p = mu
    keep = (p > 0) & (p < 1)
    X = X[:, keep]; p = p[keep]
    X = (X - p) / np.sqrt(p * (1 - p))
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    k = min(n_components, len(S))
    var = S ** 2 / np.sum(S ** 2)
    return U[:, :k] * S[:k], var[:k]


def group_association(G: np.ndarray, groups: np.ndarray, group_names: Sequence[str]) -> pd.DataFrame:
    """Per-site Fisher's exact test of allele carriage vs group (haploid case-control
    style scan), with Benjamini-Hochberg FDR. Ignores population structure, so treat
    hits as candidates and confirm with a structure-aware model (e.g. GEMMA, GAPIT)."""
    from scipy.stats import fisher_exact
    a, b = group_names[:2]
    pvals, ors = [], []
    for row in G:
        ok = ~np.isnan(row)
        x, g = row[ok] > 0, groups[ok]
        tab = [[np.sum(x & (g == b)), np.sum(~x & (g == b))],
               [np.sum(x & (g == a)), np.sum(~x & (g == a))]]
        o, p = fisher_exact(tab)
        ors.append(o); pvals.append(p)
    pvals = np.array(pvals)
    order = np.argsort(pvals)
    m = len(pvals)
    q = np.empty(m)
    q[order] = np.minimum.accumulate((pvals[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    return pd.DataFrame({"odds_ratio": ors, "pvalue": pvals, "padj": np.minimum(q, 1)})


def site_table(variants: Sequence[Variant], G: np.ndarray, groups: np.ndarray,
               group_names: List[str]) -> pd.DataFrame:
    freqs, ns = group_frequencies(G, groups, group_names)
    df = pd.DataFrame({"chrom": [v.chrom for v in variants], "pos": [v.pos for v in variants],
                       "ref": [v.ref for v in variants], "alt": [v.alts[0] for v in variants]})
    for g in group_names:
        df[f"af_{g}"] = freqs[g]; df[f"n_{g}"] = ns[g]
        df[f"pi_{g}"] = site_pi(freqs[g], ns[g])
    if len(group_names) == 2:
        a, b = group_names
        df["fst_num"], df["fst_den"] = hudson_fst(freqs[a], freqs[b], ns[a], ns[b])
        with np.errstate(invalid="ignore", divide="ignore"):
            df["fst_site"] = df.fst_num / df.fst_den
        df["delta_af"] = df[f"af_{b}"] - df[f"af_{a}"]
    return df
