import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cgpipe.annotate import (annotate_variants, cds_sequence, genomic_to_cds, revcomp,  # noqa: E402
                             screen_known_mutations, translate)
from cgpipe.cli import load_config, run  # noqa: E402
from cgpipe.filters import FilterParams, apply_filters  # noqa: E402
from cgpipe.io import Gene, Variant, parse_gt, read_vcf  # noqa: E402
from cgpipe.popgen import (hudson_fst, pairwise_distance, pca, ratio_of_averages,  # noqa: E402
                           site_pi)
from cgpipe.simulate import simulate  # noqa: E402
from cgpipe.sv import overlap_genes  # noqa: E402


def var(chrom="c", pos=1, ref="A", alt="G", dos=(0, 1), qual=50.0, info=None, depth=None):
    dos = np.array(dos, float)
    return Variant(chrom, pos, ".", ref, alt.split(","), qual, "PASS", info or {}, dos,
                   np.where(np.isnan(dos), 0, 1), np.array(depth if depth else [30] * len(dos), float))


# ---------------------------------------------------------------- parsing
def test_parse_gt_haploid_diploid_missing():
    assert parse_gt("1") == (1.0, 1)
    assert parse_gt("0/1") == (1.0, 2)
    assert parse_gt("1|1") == (2.0, 2)
    d, p = parse_gt("./.")
    assert np.isnan(d) and p == 0


def test_read_vcf_and_sv_properties(tmp_path):
    vcf = tmp_path / "t.vcf"
    vcf.write_text("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ta\tb\n"
                   "c\t10\tv1\tA\tT\t50\tPASS\t.\tGT:DP\t0:20\t1:3\n"
                   "c\t100\tv2\tN\t<DEL>\t40\tPASS\tSVTYPE=DEL;SVLEN=-500;END=599\tGT\t1\t.\n")
    samples, recs = read_vcf(str(vcf))
    recs = list(recs)
    assert samples == ["a", "b"]
    assert recs[0].is_snp and recs[0].depth[1] == 3
    assert recs[1].is_sv and recs[1].svtype == "DEL" and recs[1].svlen == 500 and recs[1].end == 599
    assert np.isnan(recs[1].dosage[1])


# ---------------------------------------------------------------- filters
def test_filters_audit_and_depth_masking():
    vs = [var(qual=10), var(alt="G,T", dos=(1, 2)), var(dos=(np.nan, np.nan)), var(dos=(0, 0)),
          var(dos=(1, 0), depth=[30, 2]), var()]
    kept, audit = apply_filters(vs, FilterParams(min_qual=30, min_dp=5, max_missing=0.5))
    assert audit["removed_QUAL"] == 1
    assert audit["removed_multiallelic"] == 1
    assert audit["removed_missingness"] == 1     # the all-missing site
    assert audit["removed_monomorphic_ref"] == 1
    assert audit["kept_snp"] == 2                # low-depth call masked: 1/2 missing is not > 0.5


def test_low_depth_genotype_becomes_missing():
    v = var(dos=(1, 1, 0, 0), depth=[30, 2, 30, 30])
    kept, _ = apply_filters([v], FilterParams(min_dp=5, max_missing=0.5))
    assert np.isnan(kept[0].dosage[1]) and kept[0].ploidy[1] == 0


# ---------------------------------------------------------------- popgen
def test_hudson_fst_extremes_and_ratio_of_averages():
    num, den = hudson_fst([0.0, 0.5], [1.0, 0.5], [10, 10], [10, 10])
    assert num[0] / den[0] == pytest.approx(1.0)
    assert num[1] / den[1] < 0                       # identical freqs -> slightly negative (unbiased)
    assert ratio_of_averages(num, den) == pytest.approx((num.sum()) / den.sum())


def test_site_pi_unbiased():
    assert site_pi(np.array([0.5]), np.array([2]))[0] == pytest.approx(1.0)
    assert site_pi(np.array([0.0]), np.array([10]))[0] == 0


def test_pairwise_distance_ignores_missing():
    G = np.array([[0, 1, 0], [1, 1, np.nan], [0, 0, 0]], float)
    D = pairwise_distance(G)
    assert D[0, 1] == pytest.approx(1 / 3)
    assert D[0, 2] == pytest.approx(0.0)             # site 2 missing in sample 2: 2 sites compared
    assert D[1, 2] == pytest.approx(1 / 2)
    assert np.allclose(D, D.T)


def test_pca_separates_two_populations():
    rng = np.random.default_rng(0)
    pa, pb = rng.uniform(0, 0.3, 300), rng.uniform(0.7, 1, 300)
    G = np.hstack([(rng.random((300, 10)) < pa[:, None]), (rng.random((300, 10)) < pb[:, None])]).astype(float)
    scores, var_ = pca(G, 2)
    assert np.sign(scores[:10, 0]).sum() * np.sign(scores[10:, 0]).sum() < 0
    assert var_[0] > 0.3


# ---------------------------------------------------------------- annotation
def test_translate_and_revcomp():
    assert translate("ATGGGTTAA") == "MG*"
    assert revcomp("ATGC") == "GCAT"


def test_minus_strand_multi_exon_coordinates():
    # gene on minus strand with two exons: 101-110 and 201-205 (genomic)
    g = Gene("g", "G", "c", 101, 205, "-", [(101, 110), (201, 205)])
    assert genomic_to_cds(g, 205) == 0
    assert genomic_to_cds(g, 201) == 4
    assert genomic_to_cds(g, 110) == 5
    assert genomic_to_cds(g, 150) is None


def test_annotation_finds_missense_on_both_strands():
    coding = "ATG" + "GGT" + "TAT" + "TAA"         # M G Y *
    genome = {"c": "CC" + coding + "CC" + revcomp(coding) + "CC"}
    plus = Gene("p", "PLUS", "c", 3, 14, "+", [(3, 14)])
    minus = Gene("m", "MINUS", "c", 17, 28, "-", [(17, 28)])
    assert translate(cds_sequence(minus, genome)) == "MGY*"
    v1 = var("c", 3 + 4, genome["c"][6], "C")       # GGT -> GCT : G2A on plus strand
    # Y3F on minus strand: coding offset 7 (2nd base of codon 3) -> genomic 28 - 7 = 21
    v2 = var("c", 21, genome["c"][20], revcomp("T"))
    ann = annotate_variants([v1, v2], {"p": plus, "m": minus}, genome)
    assert set(ann.protein_change) == {"G2A", "Y3F"}
    panel = pd.DataFrame({"gene": ["PLUS", "MINUS"], "mutation": ["G2A/S", "Y3F"]})
    hits, geno = screen_known_mutations(ann, [v1, v2], ["s1", "s2"], panel)
    assert len(hits) == 2 and geno["PLUS:G2A/S"].tolist() == [0, 1]


def test_panel_reference_mismatch_raises():
    coding = "ATGGGTTAA"
    genome = {"c": coding}
    g = Gene("p", "P", "c", 1, 9, "+", [(1, 9)])
    v = var("c", 5, "G", "C")
    ann = annotate_variants([v], {"p": g}, genome)
    with pytest.raises(ValueError, match="disagrees"):
        screen_known_mutations(ann, [v], ["a", "b"], pd.DataFrame({"gene": ["P"], "mutation": ["A2G"]}))


def test_sv_promoter_overlap_is_strand_aware():
    genes = {"m": Gene("m", "M", "c", 1000, 2000, "-", [(1000, 2000)])}
    svs = pd.DataFrame([{"id": "i1", "chrom": "c", "pos": 2300, "end": 2300, "svtype": "INS", "svlen": 100},
                        {"id": "i2", "chrom": "c", "pos": 700, "end": 700, "svtype": "INS", "svlen": 100}])
    ov = overlap_genes(svs, genes, promoter_bp=500)
    assert ov.sv_id.tolist() == ["i1"] and ov.region.iloc[0] == "promoter"


# ---------------------------------------------------------------- end to end
def test_end_to_end_recovers_planted_truth(tmp_path):
    data = tmp_path / "data"
    simulate(str(data), seed=7)
    cfg_path = tmp_path / "cfg.yaml"
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path.write_text(f"""vcf: {data}/isolates.vcf
reference: {data}/reference.fasta
gff: {data}/genes.gff3
groups: {data}/groups.tsv
panel: {here}/config/known_mutations.tsv
outdir: {tmp_path}/out
plots: false
""")
    s = run(load_config(str(cfg_path)))
    truth = pd.read_csv(data / "ground_truth.tsv", sep="\t")
    found = {(h["gene"], h["observed"]): h["n_carriers"] for h in s["known_mutations_detected"]}
    for _, t in truth[truth.feature.isin(["CYTB", "SDHB", "CYP51"])].iterrows():
        assert found[(t.feature, t.change)] == t.n_carriers
    regions = {(o["gene"], o["region"], o["svtype"]) for o in s["sv_gene_overlaps"]}
    assert ("CYP51", "promoter", "INS") in regions and ("ABC1", "gene_body", "DUP") in regions
    w = pd.read_csv(tmp_path / "out" / "windows.tsv", sep="\t")
    top = w.sort_values("fst_hudson", ascending=False).iloc[0]
    assert top.chrom == "contig_1" and 50_000 <= top.start <= 65_000   # planted differentiated block
