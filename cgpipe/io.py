"""Readers for FASTA, GFF3 and VCF (plain or gzipped), with no compiled dependencies.

The VCF reader is deliberately small and explicit: it keeps the fields this
pipeline needs (site info, genotypes, per-sample depth) and turns genotypes into
alternate-allele dosages, which handles haploid fungi and diploid samples alike.
"""
from __future__ import annotations

import gzip
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

import numpy as np


def _open(path: str):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


# --------------------------------------------------------------------------- FASTA
def read_fasta(path: str) -> Dict[str, str]:
    seqs: Dict[str, List[str]] = {}
    name = None
    with _open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                name = line[1:].split()[0]
                if name in seqs:
                    raise ValueError(f"Duplicated FASTA record: {name}")
                seqs[name] = []
            elif name is None:
                raise ValueError("FASTA sequence found before first header")
            else:
                seqs[name].append(line.upper())
    return {k: "".join(v) for k, v in seqs.items()}


# --------------------------------------------------------------------------- GFF3
@dataclass
class Gene:
    gene_id: str
    name: str
    chrom: str
    start: int          # 1-based inclusive
    end: int            # 1-based inclusive
    strand: str
    cds: List[tuple] = field(default_factory=list)   # [(start, end), ...] 1-based inclusive

    @property
    def cds_sorted(self) -> List[tuple]:
        """CDS segments in transcription order."""
        return sorted(self.cds, key=lambda x: x[0], reverse=(self.strand == "-"))


def _attrs(s: str) -> Dict[str, str]:
    out = {}
    for kv in s.strip().strip(";").split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def read_gff(path: str) -> Dict[str, Gene]:
    """Read genes and their CDS segments (gene -> mRNA -> CDS or gene -> CDS)."""
    genes: Dict[str, Gene] = {}
    parent_of: Dict[str, str] = {}
    cds_rows = []
    with _open(path) as fh:
        for line in fh:
            if line.startswith("##FASTA"):
                break
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) != 9:
                continue
            chrom, _, ftype, start, end, _, strand, _, attr = f
            a = _attrs(attr)
            if ftype == "gene":
                gid = a["ID"]
                genes[gid] = Gene(gid, a.get("Name", gid), chrom, int(start), int(end), strand)
            elif ftype in ("mRNA", "transcript"):
                parent_of[a["ID"]] = a["Parent"]
            elif ftype == "CDS":
                cds_rows.append((a["Parent"], int(start), int(end)))
    for parent, s, e in cds_rows:
        gid = parent_of.get(parent, parent)
        if gid not in genes:
            raise ValueError(f"CDS parent {parent} has no gene")
        genes[gid].cds.append((s, e))
    return genes


# --------------------------------------------------------------------------- VCF
@dataclass
class Variant:
    chrom: str
    pos: int
    vid: str
    ref: str
    alts: List[str]
    qual: Optional[float]
    filt: str
    info: Dict[str, str]
    dosage: np.ndarray          # alt-allele count per sample (float, NaN = missing)
    ploidy: np.ndarray          # ploidy per sample (0 when missing)
    depth: np.ndarray           # FORMAT/DP per sample (NaN if absent)

    @property
    def is_sv(self) -> bool:
        return "SVTYPE" in self.info or any(a.startswith("<") for a in self.alts)

    @property
    def is_snp(self) -> bool:
        return (not self.is_sv) and len(self.ref) == 1 and all(len(a) == 1 for a in self.alts)

    @property
    def is_biallelic(self) -> bool:
        return len(self.alts) == 1

    @property
    def svtype(self) -> Optional[str]:
        if "SVTYPE" in self.info:
            return self.info["SVTYPE"]
        if self.alts and self.alts[0].startswith("<"):
            return self.alts[0].strip("<>").split(":")[0]
        return None

    @property
    def end(self) -> int:
        if "END" in self.info:
            return int(self.info["END"])
        if "SVLEN" in self.info and self.svtype != "INS":
            return self.pos + abs(int(self.info["SVLEN"])) - 1
        return self.pos + len(self.ref) - 1

    @property
    def svlen(self) -> Optional[int]:
        if "SVLEN" in self.info:
            return abs(int(self.info["SVLEN"]))
        if self.is_sv:
            return self.end - self.pos + 1
        return None


def parse_gt(gt: str):
    """Return (alt dosage, ploidy). Any non-zero allele counts as alternate."""
    if gt in (".", "./.", ".|."):
        return np.nan, 0
    alleles = gt.replace("|", "/").split("/")
    if any(a == "." for a in alleles):
        return np.nan, 0
    return float(sum(a != "0" for a in alleles)), len(alleles)


def read_vcf(path: str) -> tuple[List[str], Iterator[Variant]]:
    """Return sample names and an iterator over variants."""
    fh = _open(path)
    samples: List[str] = []
    for line in fh:
        if line.startswith("##"):
            continue
        if line.startswith("#CHROM"):
            samples = line.rstrip("\n").split("\t")[9:]
            break
    else:
        fh.close()
        raise ValueError("VCF has no #CHROM header line")

    def records() -> Iterator[Variant]:
        try:
            for line in fh:
                if not line.strip() or line.startswith("#"):
                    continue
                f = line.rstrip("\n").split("\t")
                info = {}
                if f[7] != ".":
                    for kv in f[7].split(";"):
                        k, _, v = kv.partition("=")
                        info[k] = v if v else "true"
                fmt = f[8].split(":") if len(f) > 8 else []
                gi = fmt.index("GT") if "GT" in fmt else None
                di = fmt.index("DP") if "DP" in fmt else None
                n = len(samples)
                dos = np.full(n, np.nan); pl = np.zeros(n, dtype=int); dp = np.full(n, np.nan)
                for j, s in enumerate(f[9:9 + n]):
                    parts = s.split(":")
                    if gi is not None and gi < len(parts):
                        dos[j], pl[j] = parse_gt(parts[gi])
                    if di is not None and di < len(parts) and parts[di] not in (".", ""):
                        dp[j] = float(parts[di])
                yield Variant(f[0], int(f[1]), f[2], f[3].upper(), f[4].upper().split(","),
                              None if f[5] == "." else float(f[5]), f[6], info, dos, pl, dp)
        finally:
            fh.close()

    return samples, records()


def read_groups(path: str) -> Dict[str, str]:
    """Two-column TSV: sample<TAB>group (header optional, '#' comments allowed)."""
    out = {}
    with open(path) as fh:
        for i, line in enumerate(fh):
            if not line.strip() or line.startswith("#"):
                continue
            s, g = line.rstrip("\n").split("\t")[:2]
            if i == 0 and s.lower() in ("sample", "sample_id", "isolate"):
                continue
            out[s] = g
    return out
