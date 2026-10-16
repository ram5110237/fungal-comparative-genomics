"""Site- and genotype-level filters with a per-reason audit trail."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np

from .io import Variant


@dataclass
class FilterParams:
    min_qual: float = 30.0
    min_dp: float = 5.0              # per-genotype depth; lower calls are set to missing
    max_missing: float = 0.2         # maximum fraction of missing genotypes per site
    min_maf: float = 0.0             # minor allele frequency (SNPs only)
    biallelic_only: bool = True
    pass_only: bool = True


def allele_frequency(v: Variant) -> float:
    ok = ~np.isnan(v.dosage)
    if not ok.any():
        return np.nan
    return float(v.dosage[ok].sum() / v.ploidy[ok].sum())


def mask_low_depth(v: Variant, min_dp: float) -> Variant:
    if min_dp > 0:
        low = ~np.isnan(v.depth) & (v.depth < min_dp)
        v.dosage = v.dosage.copy(); v.ploidy = v.ploidy.copy()
        v.dosage[low] = np.nan; v.ploidy[low] = 0
    return v


def site_failure(v: Variant, p: FilterParams) -> str | None:
    """Return the first failing reason, or None if the site passes."""
    if p.pass_only and v.filt not in ("PASS", "."):
        return "FILTER"
    if v.qual is not None and v.qual < p.min_qual:
        return "QUAL"
    if p.biallelic_only and not v.is_biallelic:
        return "multiallelic"
    if np.mean(np.isnan(v.dosage)) > p.max_missing:
        return "missingness"
    if v.is_snp and p.min_maf > 0:
        af = allele_frequency(v)
        if np.isnan(af) or min(af, 1 - af) < p.min_maf:
            return "MAF"
    if v.is_snp or v.is_sv:
        af = allele_frequency(v)
        if not np.isnan(af) and af == 0:
            return "monomorphic_ref"
    return None


def apply_filters(variants: Iterable[Variant], p: FilterParams) -> Tuple[List[Variant], Counter]:
    kept, audit = [], Counter()
    for v in variants:
        audit["input"] += 1
        v = mask_low_depth(v, p.min_dp)
        reason = site_failure(v, p)
        if reason:
            audit[f"removed_{reason}"] += 1
        else:
            kept.append(v)
            audit["kept_sv" if v.is_sv else ("kept_snp" if v.is_snp else "kept_indel")] += 1
    return kept, audit
