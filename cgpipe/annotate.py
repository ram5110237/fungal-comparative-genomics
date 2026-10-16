"""Coding-consequence annotation of SNPs and screening against a panel of known
target-site mutations (e.g. fungicide resistance alleles).

Handles plus- and minus-strand genes and multi-exon CDS models.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .io import Gene, Variant

_BASES = "TCAG"
_AA = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON_TABLE = {a + b + c: _AA[16 * i + 4 * j + k]
               for i, a in enumerate(_BASES) for j, b in enumerate(_BASES) for k, c in enumerate(_BASES)}
_COMP = str.maketrans("ACGTN", "TGCAN")

AA3 = {"A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys", "Q": "Gln", "E": "Glu", "G": "Gly",
       "H": "His", "I": "Ile", "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe", "P": "Pro", "S": "Ser",
       "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val", "*": "Ter"}


def revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def translate(seq: str) -> str:
    return "".join(CODON_TABLE.get(seq[i:i + 3], "X") for i in range(0, len(seq) - 2, 3))


def cds_sequence(gene: Gene, genome: Dict[str, str]) -> str:
    chrom = genome[gene.chrom]
    parts = [chrom[s - 1:e] for s, e in sorted(gene.cds)]
    seq = "".join(parts)
    return revcomp(seq) if gene.strand == "-" else seq


def genomic_to_cds(gene: Gene, pos: int) -> Optional[int]:
    """0-based offset of a genomic position within the spliced CDS (in transcription
    direction), or None if the position is not coding."""
    offset = 0
    for s, e in gene.cds_sorted:
        if s <= pos <= e:
            return offset + (pos - s if gene.strand == "+" else e - pos)
        offset += e - s + 1
    return None


def annotate_snp(v: Variant, genes: Sequence[Gene], genome: Dict[str, str],
                 cds_cache: Dict[str, str]) -> List[dict]:
    out = []
    for g in genes:
        if g.chrom != v.chrom or not (g.start <= v.pos <= g.end):
            continue
        off = genomic_to_cds(g, v.pos)
        if off is None:
            out.append({"gene": g.name, "consequence": "intronic_or_utr"})
            continue
        cds = cds_cache.setdefault(g.gene_id, cds_sequence(g, genome))
        ref_base = genome[v.chrom][v.pos - 1]
        if ref_base != v.ref:
            raise ValueError(f"REF mismatch at {v.chrom}:{v.pos}: VCF {v.ref}, genome {ref_base}")
        alt = v.alts[0] if g.strand == "+" else revcomp(v.alts[0])
        codon_idx, within = divmod(off, 3)
        ref_codon = cds[3 * codon_idx:3 * codon_idx + 3]
        alt_codon = ref_codon[:within] + alt + ref_codon[within + 1:]
        ra, aa = CODON_TABLE.get(ref_codon, "X"), CODON_TABLE.get(alt_codon, "X")
        cons = ("synonymous" if ra == aa else "stop_gained" if aa == "*"
                else "stop_lost" if ra == "*" else "missense")
        out.append({"gene": g.name, "consequence": cons, "codon": codon_idx + 1,
                    "ref_codon": ref_codon, "alt_codon": alt_codon,
                    "protein_change": f"{ra}{codon_idx + 1}{aa}",
                    "hgvs_p": f"p.{AA3.get(ra, 'Xaa')}{codon_idx + 1}{AA3.get(aa, 'Xaa')}"})
    return out


def annotate_variants(variants: Sequence[Variant], genes: Dict[str, Gene],
                      genome: Dict[str, str]) -> pd.DataFrame:
    gl = list(genes.values())
    cache: Dict[str, str] = {}
    rows = []
    for i, v in enumerate(variants):
        if not v.is_snp:
            continue
        for a in annotate_snp(v, gl, genome, cache):
            rows.append({"site_index": i, "chrom": v.chrom, "pos": v.pos, "ref": v.ref,
                         "alt": v.alts[0], **a})
    cols = ["site_index", "chrom", "pos", "ref", "alt", "gene", "consequence", "codon",
            "ref_codon", "alt_codon", "protein_change", "hgvs_p"]
    return pd.DataFrame(rows, columns=cols)


def screen_known_mutations(annot: pd.DataFrame, variants: Sequence[Variant], samples: Sequence[str],
                           panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match annotated missense changes against a panel (columns: gene, mutation,
    and optional fungicide_group / note). Returns (hits, per-sample genotype matrix).

    A panel entry such as G143A matches any alt amino acid listed after the
    position, so 'H272R/Y' matches H272R and H272Y.
    """
    hits, geno = [], pd.DataFrame(index=list(samples))
    for _, p in panel.iterrows():
        mut = str(p["mutation"])
        ref_aa, pos, alts = mut[0], int("".join(c for c in mut if c.isdigit())), \
            mut[len(mut[0] + "".join(c for c in mut if c.isdigit())):].split("/")
        m = annot[(annot.gene == p["gene"]) & (annot.codon == pos)]
        col = f"{p['gene']}:{mut}"
        calls = np.zeros(len(samples))
        found = False
        for _, a in m.iterrows():
            obs = a["protein_change"]
            if obs[0] != ref_aa:
                raise ValueError(f"Panel reference {ref_aa}{pos} disagrees with annotation {obs} "
                                 f"in {p['gene']}: check gene model or panel numbering")
            if obs[-1] in alts:
                v = variants[int(a["site_index"])]
                carrier = np.nan_to_num(v.dosage, nan=0) > 0
                calls = np.maximum(calls, carrier.astype(float))
                missing = np.isnan(v.dosage)
                found = True
                hits.append({"gene": p["gene"], "panel_mutation": mut, "observed": obs,
                             "chrom": a["chrom"], "pos": a["pos"],
                             "n_carriers": int(carrier.sum()), "n_missing": int(missing.sum()),
                             "carriers": ",".join(s for s, c in zip(samples, carrier) if c),
                             **{k: p[k] for k in panel.columns if k not in ("gene", "mutation")}})
        geno[col] = calls.astype(int) if found else 0
    return pd.DataFrame(hits), geno
