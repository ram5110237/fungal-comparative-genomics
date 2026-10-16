"""Synthetic example data: a two-contig fungal reference, gene models, and a VCF of
24 haploid isolates (12 fungicide-sensitive, 12 resistant) with planted target-site
mutations, a differentiated genomic region, structural variants and messy records
(low quality, multiallelic, missing and low-depth genotypes) to exercise the filters.

Everything here is artificial. Gene names mimic common fungicide targets only so
that the example reads naturally.
"""
from __future__ import annotations

import os

import numpy as np

from .annotate import CODON_TABLE, revcomp

SENSE = [c for c, a in CODON_TABLE.items() if a != "*"]


def _coding_seq(rng, n_codons, forced):
    codons = [SENSE[i] for i in rng.integers(0, len(SENSE), n_codons)]
    codons[0] = "ATG"; codons[-1] = "TAA"
    for pos, c in forced.items():
        codons[pos - 1] = c
    return "".join(codons)


def simulate(outdir: str, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    os.makedirs(outdir, exist_ok=True)
    lengths = {"contig_1": 80_000, "contig_2": 50_000}
    genome = {c: list(rng.choice(list("ACGT"), n)) for c, n in lengths.items()}

    # ---- gene models -------------------------------------------------------
    # CYTB: + strand, single exon, 400 codons at contig_1:20001-21200
    cytb = _coding_seq(rng, 400, {129: "TTC", 143: "GGT"})
    genome["contig_1"][20000:21200] = list(cytb)
    # SDHB: + strand, two exons (150 + 150 codons) with an 80 bp intron
    sdhb = _coding_seq(rng, 300, {272: "CAT"})
    genome["contig_1"][40000:40450] = list(sdhb[:450])
    genome["contig_1"][40450:40452] = list("GT"); genome["contig_1"][40528:40530] = list("AG")
    genome["contig_1"][40530:40980] = list(sdhb[450:])
    # CYP51: - strand, 550 codons at contig_2:15001-16650
    cyp51 = _coding_seq(rng, 550, {137: "TAT", 524: "TCT"})
    genome["contig_2"][15000:16650] = list(revcomp(cyp51))
    # ABC1: + strand, contig_2:30001-32000 (target of a duplication)
    abc1 = _coding_seq(rng, 666, {})
    genome["contig_2"][30000:31998] = list(abc1)

    gff = ["##gff-version 3",
           "contig_1\tsim\tgene\t20001\t21200\t.\t+\t.\tID=gene_cytb;Name=CYTB",
           "contig_1\tsim\tmRNA\t20001\t21200\t.\t+\t.\tID=mrna_cytb;Parent=gene_cytb",
           "contig_1\tsim\tCDS\t20001\t21200\t.\t+\t0\tID=cds_cytb;Parent=mrna_cytb",
           "contig_1\tsim\tgene\t40001\t40980\t.\t+\t.\tID=gene_sdhb;Name=SDHB",
           "contig_1\tsim\tmRNA\t40001\t40980\t.\t+\t.\tID=mrna_sdhb;Parent=gene_sdhb",
           "contig_1\tsim\tCDS\t40001\t40450\t.\t+\t0\tID=cds_sdhb;Parent=mrna_sdhb",
           "contig_1\tsim\tCDS\t40531\t40980\t.\t+\t0\tID=cds_sdhb;Parent=mrna_sdhb",
           "contig_2\tsim\tgene\t15001\t16650\t.\t-\t.\tID=gene_cyp51;Name=CYP51",
           "contig_2\tsim\tmRNA\t15001\t16650\t.\t-\t.\tID=mrna_cyp51;Parent=gene_cyp51",
           "contig_2\tsim\tCDS\t15001\t16650\t.\t-\t0\tID=cds_cyp51;Parent=mrna_cyp51",
           "contig_2\tsim\tgene\t30001\t31998\t.\t+\t.\tID=gene_abc1;Name=ABC1",
           "contig_2\tsim\tCDS\t30001\t31998\t.\t+\t0\tID=cds_abc1;Parent=gene_abc1"]

    samples = [f"S{i:02d}" for i in range(1, 13)] + [f"R{i:02d}" for i in range(1, 13)]
    is_r = np.array([s.startswith("R") for s in samples])
    n = len(samples)

    def carriers(names):
        return np.array([s in names for s in samples], dtype=float)

    records = []   # (chrom, pos, ref, alt, qual, info, dosage, filter)

    def add(chrom, pos, alt, dos, qual=60.0, info=".", ref=None, filt="PASS"):
        r = ref if ref is not None else genome[chrom][pos - 1]
        records.append((chrom, pos, r, alt, qual, info, np.asarray(dos, float), filt))

    # ---- planted target-site mutations --------------------------------------
    truth = []
    g143a = carriers({f"R{i:02d}" for i in range(1, 10)})
    add("contig_1", 20000 + 142 * 3 + 2, "C", g143a)            # GGT -> GCT, G143A
    truth.append(("CYTB", "G143A", int(g143a.sum())))
    h272r = carriers({"R02", "R04", "R06", "R08", "R10"})
    add("contig_1", 40531 + (271 * 3 - 450) + 1, "G", h272r)      # CAT -> CGT, H272R (exon 2)
    truth.append(("SDHB", "H272R", int(h272r.sum())))
    y137f = carriers({"R01", "R03", "R05", "R07", "R09", "R11", "R12", "S05"})
    p = 16650 - (136 * 3 + 1)                                    # coding A -> T on minus strand
    add("contig_2", p, "A", y137f)                               # genomic T -> A
    truth.append(("CYP51", "Y137F", int(y137f.sum())))
    reserved = {("contig_1", 20000 + 142 * 3 + 2), ("contig_1", 40531 + (271 * 3 - 450) + 1), ("contig_2", p)}

    # ---- background SNPs -----------------------------------------------------
    for chrom, length in lengths.items():
        n_snps = int(length / 80)
        for pos in sorted(rng.choice(np.arange(100, length - 100), n_snps, replace=False)):
            if (chrom, int(pos)) in reserved:
                continue
            p0 = rng.beta(0.6, 1.2)
            if chrom == "contig_1" and 55_000 <= pos <= 65_000:          # differentiated block
                pa, pb = rng.uniform(0.0, 0.2), rng.uniform(0.8, 1.0)
            else:
                pa = np.clip(p0 + rng.normal(0, 0.08), 0, 1); pb = np.clip(p0 + rng.normal(0, 0.08), 0, 1)
            dos = np.where(is_r, rng.random(n) < pb, rng.random(n) < pa).astype(float)
            ref = genome[chrom][pos - 1]
            alt = rng.choice([b for b in "ACGT" if b != ref])
            add(chrom, int(pos), alt, dos)

    # ---- messy records -------------------------------------------------------
    for _ in range(25):  # low quality
        c = rng.choice(list(lengths)); pos = int(rng.integers(100, lengths[c] - 100))
        ref = genome[c][pos - 1]
        add(c, pos, rng.choice([b for b in "ACGT" if b != ref]), rng.random(n) < 0.3, qual=12.0)
    for _ in range(10):  # multiallelic
        c = rng.choice(list(lengths)); pos = int(rng.integers(100, lengths[c] - 100))
        ref = genome[c][pos - 1]
        alts = [b for b in "ACGT" if b != ref][:2]
        add(c, pos, ",".join(alts), rng.integers(0, 3, n))
    for _ in range(40):  # small indels
        c = rng.choice(list(lengths)); pos = int(rng.integers(100, lengths[c] - 100))
        ref = genome[c][pos - 1]
        if rng.random() < 0.5:
            add(c, pos, ref + "".join(rng.choice(list("ACGT"), int(rng.integers(1, 6)))), rng.random(n) < 0.4)
        else:
            k = int(rng.integers(1, 6)); r = "".join(genome[c][pos - 1:pos + k])
            add(c, pos, ref, rng.random(n) < 0.4, ref=r)

    # ---- structural variants ---------------------------------------------------
    ins = carriers({f"R{i:02d}" for i in (1, 2, 3, 5, 7, 9, 11, 12)})
    add("contig_2", 16900, "<INS>", ins, info="SVTYPE=INS;SVLEN=150;END=16900")   # CYP51 promoter
    dup = carriers({"R02", "R04", "R06", "R08", "R10", "R12"})
    add("contig_2", 29500, "<DUP>", dup, info="SVTYPE=DUP;SVLEN=3001;END=32500")  # whole ABC1
    truth += [("CYP51_promoter", "INS150", int(ins.sum())), ("ABC1", "DUP", int(dup.sum()))]
    for _ in range(30):
        c = rng.choice(list(lengths)); t = rng.choice(["DEL", "INS", "INV", "DUP"])
        size = int(np.clip(rng.lognormal(6.5, 1.3), 50, 20_000))
        pos = int(rng.integers(500, lengths[c] - size - 500))
        if c == "contig_2" and 14_000 < pos < 33_000:
            continue
        end = pos if t == "INS" else pos + size - 1
        svlen = -size if t == "DEL" else size
        add(c, pos, f"<{t}>", rng.random(n) < rng.uniform(0.05, 0.5),
            info=f"SVTYPE={t};SVLEN={svlen};END={end}", qual=float(rng.uniform(20, 99)))

    # ---- genotype missingness and depth ---------------------------------------
    records.sort(key=lambda r: (r[0], r[1]))
    vcf_lines = ["##fileformat=VCFv4.2", "##source=cgpipe.simulate"]
    vcf_lines += [f"##contig=<ID={c},length={n_}>" for c, n_ in lengths.items()]
    vcf_lines += ['##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">',
                  '##INFO=<ID=SVLEN,Number=1,Type=Integer,Description="Length of structural variant">',
                  '##INFO=<ID=END,Number=1,Type=Integer,Description="End position">',
                  '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
                  '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read depth">',
                  "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples)]
    for i, (c, pos, ref, alt, q, info, dos, filt) in enumerate(records):
        planted = (c, pos) in reserved or "END=16900" in info or "END=32500" in info
        gts = []
        for d in dos:
            dp = int(rng.poisson(25))
            if not planted and rng.random() < 0.02:
                gts.append(".:0"); continue
            if not planted and rng.random() < 0.02:
                dp = int(rng.integers(1, 4))
            gts.append(f"{int(d)}:{dp}")
        vcf_lines.append("\t".join([c, str(pos), f"var{i + 1}", ref, alt, f"{q:.1f}", filt, info, "GT:DP"] + gts))

    with open(os.path.join(outdir, "reference.fasta"), "w") as fh:
        for c, s in genome.items():
            seq = "".join(s)
            fh.write(f">{c}\n" + "\n".join(seq[i:i + 80] for i in range(0, len(seq), 80)) + "\n")
    with open(os.path.join(outdir, "genes.gff3"), "w") as fh:
        fh.write("\n".join(gff) + "\n")
    with open(os.path.join(outdir, "isolates.vcf"), "w") as fh:
        fh.write("\n".join(vcf_lines) + "\n")
    with open(os.path.join(outdir, "groups.tsv"), "w") as fh:
        fh.write("sample\tgroup\n" + "".join(f"{s}\t{'resistant' if s.startswith('R') else 'sensitive'}\n"
                                             for s in samples))
    with open(os.path.join(outdir, "ground_truth.tsv"), "w") as fh:
        fh.write("feature\tchange\tn_carriers\n" + "".join(f"{a}\t{b}\t{c}\n" for a, b, c in truth))
    return {"n_records": len(records), "samples": samples, "truth": truth}
