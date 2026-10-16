"""Structural-variant summaries and gene/promoter overlap."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd

from .io import Gene, Variant

SIZE_BINS = [0, 100, 1_000, 10_000, 100_000, np.inf]
SIZE_LABELS = ["50-100 bp", "100 bp-1 kb", "1-10 kb", "10-100 kb", ">100 kb"]


def sv_table(variants: Sequence[Variant], samples: Sequence[str], groups: Dict[str, str]) -> pd.DataFrame:
    gnames = sorted(set(groups.values()))
    garr = np.array([groups.get(s, "NA") for s in samples])
    rows = []
    for v in variants:
        if not v.is_sv:
            continue
        carrier = np.nan_to_num(v.dosage, nan=0) > 0
        row = {"chrom": v.chrom, "pos": v.pos, "end": v.end, "svtype": v.svtype,
               "svlen": v.svlen, "id": v.vid, "n_carriers": int(carrier.sum())}
        for g in gnames:
            sel = garr == g
            called = sel & ~np.isnan(v.dosage)
            row[f"carrier_freq_{g}"] = float(carrier[called].mean()) if called.any() else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df["size_class"] = pd.cut(df.svlen, SIZE_BINS, labels=SIZE_LABELS, right=False).astype(str)
    return df


def overlap_genes(svs: pd.DataFrame, genes: Dict[str, Gene], promoter_bp: int = 1000) -> pd.DataFrame:
    """Report each SV-gene pair where the SV overlaps the gene body or its promoter
    (promoter_bp upstream of the gene start, strand-aware)."""
    rows = []
    for _, s in svs.iterrows():
        # insertions are point events: use the breakpoint
        s_start, s_end = (s.pos, s.pos) if s.svtype == "INS" else (s.pos, s.end)
        for g in genes.values():
            if g.chrom != s.chrom:
                continue
            if g.strand == "+":
                prom = (max(1, g.start - promoter_bp), g.start - 1)
            else:
                prom = (g.end + 1, g.end + promoter_bp)
            body = s_start <= g.end and s_end >= g.start
            in_prom = s_start <= prom[1] and s_end >= prom[0]
            if body or in_prom:
                full = s_start <= g.start and s_end >= g.end and s.svtype != "INS"
                rows.append({"sv_id": s.id, "svtype": s.svtype, "svlen": s.svlen,
                             "gene": g.name, "region": "gene_body" if body else "promoter",
                             "whole_gene": bool(full),
                             **{k: s[k] for k in svs.columns if k.startswith("carrier_freq_")}})
    return pd.DataFrame(rows)


def sv_summary(svs: pd.DataFrame) -> pd.DataFrame:
    if not len(svs):
        return pd.DataFrame(columns=["svtype", "size_class", "count"])
    return (svs.groupby(["svtype", "size_class"]).size().reset_index(name="count")
            .sort_values(["svtype", "size_class"]))
