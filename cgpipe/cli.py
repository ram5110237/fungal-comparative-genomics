"""Command-line entry point.

    cgpipe simulate -o data/example
    cgpipe run -c config/config.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml

from . import __version__
from .annotate import annotate_variants, screen_known_mutations
from .filters import FilterParams, apply_filters
from .io import read_fasta, read_gff, read_groups, read_vcf
from .popgen import (genotype_matrix, group_association, pairwise_distance, pca,
                     ratio_of_averages, site_table, windowed_stats)
from .simulate import simulate
from .sv import overlap_genes, sv_summary, sv_table


def _log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, file=sys.stderr, flush=True)


def load_config(path):
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    base = os.path.dirname(os.path.abspath(path))
    for k in ("vcf", "reference", "gff", "groups", "panel", "outdir"):
        if cfg.get(k) and not os.path.isabs(cfg[k]):
            cfg[k] = os.path.normpath(os.path.join(base, cfg[k]))
    for k in ("vcf", "groups", "outdir"):
        if not cfg.get(k):
            raise ValueError(f"config is missing '{k}'")
    cfg.setdefault("filters", {})
    cfg.setdefault("window_size", 5000)
    cfg.setdefault("promoter_bp", 1000)
    cfg.setdefault("plots", True)
    return cfg


def run(cfg) -> dict:
    out = cfg["outdir"]
    os.makedirs(out, exist_ok=True)
    groups = read_groups(cfg["groups"])

    _log("Reading and filtering VCF")
    samples, recs = read_vcf(cfg["vcf"])
    missing = [s for s in samples if s not in groups]
    if missing:
        raise ValueError(f"Samples without a group assignment: {missing}")
    params = FilterParams(**cfg["filters"])
    kept, audit = apply_filters(recs, params)
    pd.DataFrame(sorted(audit.items()), columns=["step", "n"]).to_csv(
        os.path.join(out, "filter_audit.tsv"), sep="\t", index=False)

    snps = [v for v in kept if v.is_snp]
    svs = [v for v in kept if v.is_sv]
    garr = np.array([groups[s] for s in samples])
    gnames = sorted(set(garr))
    summary = {"version": __version__, "samples": len(samples), "groups": {g: int((garr == g).sum()) for g in gnames},
               "filter_audit": dict(audit), "snps": len(snps), "svs": len(svs)}

    # ---- SNP population genetics ----------------------------------------------
    _log(f"Population genetics on {len(snps)} SNPs")
    G = genotype_matrix(snps)
    st = site_table(snps, G, garr, gnames)
    if len(gnames) == 2:
        st = pd.concat([st, group_association(G, garr, gnames)], axis=1)
        summary["fst_hudson_genome"] = ratio_of_averages(st.fst_num.values, st.fst_den.values)
        summary["assoc_sites_fdr05"] = int((st.padj < 0.05).sum())
    st.to_csv(os.path.join(out, "snp_sites.tsv"), sep="\t", index=False)

    lengths = None
    genome = genes = None
    if cfg.get("reference"):
        genome = read_fasta(cfg["reference"])
        lengths = {c: len(s) for c, s in genome.items()}
    else:
        lengths = st.groupby("chrom").pos.max().to_dict()
    win = windowed_stats(st, lengths, cfg["window_size"], gnames)
    win.to_csv(os.path.join(out, "windows.tsv"), sep="\t", index=False)
    for g in gnames:
        summary[f"pi_per_bp_{g}"] = float(np.nansum(st[f"pi_{g}"]) / sum(lengths.values()))

    D = pairwise_distance(G)
    pd.DataFrame(D, index=samples, columns=samples).to_csv(os.path.join(out, "distance_matrix.tsv"), sep="\t")
    scores, var = pca(G, 5)
    pcs = pd.DataFrame(scores, columns=[f"PC{i + 1}" for i in range(scores.shape[1])])
    pcs.insert(0, "group", garr); pcs.insert(0, "sample", samples)
    pcs.to_csv(os.path.join(out, "pca.tsv"), sep="\t", index=False)
    pd.DataFrame({"pc": pcs.columns[2:], "var_explained": var}).to_csv(
        os.path.join(out, "pca_variance.tsv"), sep="\t", index=False)

    # ---- coding annotation and known-mutation screen ----------------------------
    if genome is not None and cfg.get("gff"):
        _log("Annotating coding consequences")
        genes = read_gff(cfg["gff"])
        ann = annotate_variants(snps, genes, genome)
        ann.to_csv(os.path.join(out, "snp_annotation.tsv"), sep="\t", index=False)
        summary["coding_consequences"] = ann.consequence.value_counts().to_dict()
        if cfg.get("panel"):
            panel = pd.read_csv(cfg["panel"], sep="\t", comment="#")
            hits, geno = screen_known_mutations(ann, snps, samples, panel)
            hits.to_csv(os.path.join(out, "known_mutation_hits.tsv"), sep="\t", index=False)
            geno.insert(0, "group", garr)
            geno.index.name = "sample"
            geno.to_csv(os.path.join(out, "isolate_resistance_profile.tsv"), sep="\t")
            summary["known_mutations_detected"] = (
                hits[["gene", "observed", "n_carriers"]].to_dict("records") if len(hits) else [])

    # ---- structural variants ---------------------------------------------------
    _log(f"Summarising {len(svs)} structural variants")
    svt = sv_table(svs, samples, groups)
    svt.to_csv(os.path.join(out, "sv_calls.tsv"), sep="\t", index=False)
    sv_summary(svt).to_csv(os.path.join(out, "sv_summary.tsv"), sep="\t", index=False)
    if genes is not None and len(svt):
        ov = overlap_genes(svt, genes, cfg["promoter_bp"])
        ov.to_csv(os.path.join(out, "sv_gene_overlap.tsv"), sep="\t", index=False)
        summary["sv_gene_overlaps"] = ov[["sv_id", "svtype", "gene", "region"]].to_dict("records") if len(ov) else []

    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    _write_report(summary, out)

    if cfg["plots"]:
        rscript = shutil.which("Rscript")
        script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "R", "plot_genomics.R")
        if rscript and os.path.exists(script):
            _log("Plotting with R")
            subprocess.run([rscript, script, out], check=True)
        else:
            _log("Rscript not found; skipping plots")
    _log(f"Done. Results in {out}")
    return summary


def _write_report(s, out):
    L = ["# Comparative genomics report", "",
         f"Isolates: {s['samples']} ({', '.join(f'{k}: {v}' for k, v in s['groups'].items())})", "",
         "## Filtering", "", "| step | n |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in sorted(s["filter_audit"].items())]
    L += ["", "## Population structure", ""]
    if "fst_hudson_genome" in s:
        L.append(f"Genome-wide Hudson Fst: {s['fst_hudson_genome']:.3f}. "
                 f"SNPs associated with group at FDR 5%: {s['assoc_sites_fdr05']}.")
    L.append("Nucleotide diversity per bp: " + ", ".join(f"{k[10:]} {v:.2e}" for k, v in s.items() if k.startswith("pi_per_bp_")))
    if s.get("known_mutations_detected") is not None:
        L += ["", "## Known target-site mutations", "", "| gene | change | carriers |", "|---|---|---|"]
        L += [f"| {h['gene']} | {h['observed']} | {h['n_carriers']} |" for h in s["known_mutations_detected"]]
    if s.get("sv_gene_overlaps"):
        L += ["", "## Structural variants touching genes", "", "| SV | type | gene | region |", "|---|---|---|---|"]
        L += [f"| {o['sv_id']} | {o['svtype']} | {o['gene']} | {o['region']} |" for o in s["sv_gene_overlaps"]]
    with open(os.path.join(out, "REPORT.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cgpipe", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the full pipeline from a YAML config")
    r.add_argument("-c", "--config", required=True)
    r.add_argument("--no-plots", action="store_true")
    s = sub.add_parser("simulate", help="write the synthetic example dataset")
    s.add_argument("-o", "--outdir", default="data/example")
    s.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    if a.cmd == "simulate":
        info = simulate(a.outdir, a.seed)
        _log(f"Wrote {info['n_records']} VCF records for {len(info['samples'])} isolates to {a.outdir}")
    else:
        cfg = load_config(a.config)
        if a.no_plots:
            cfg["plots"] = False
        run(cfg)


if __name__ == "__main__":
    main()
