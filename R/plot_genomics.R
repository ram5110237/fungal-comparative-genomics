#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# Figures from cgpipe output tables.
#   Rscript R/plot_genomics.R results/example
# ---------------------------------------------------------------------------
suppressPackageStartupMessages(library(ggplot2))
args <- commandArgs(trailingOnly = TRUE)
res <- if (length(args)) args[1] else "results/example"
fig <- file.path(res, "figures"); dir.create(fig, showWarnings = FALSE)
rd <- function(f) utils::read.delim(file.path(res, f), check.names = FALSE, stringsAsFactors = FALSE)
th <- theme_bw(base_size = 11) + theme(panel.grid.minor = element_blank())
save <- function(p, f, w = 7, h = 4.5) ggsave(file.path(fig, f), p, width = w, height = h, dpi = 300)

# PCA ------------------------------------------------------------------------
pcs <- rd("pca.tsv"); ve <- rd("pca_variance.tsv")
save(ggplot(pcs, aes(PC1, PC2, colour = group, label = sample)) +
       geom_point(size = 3) + geom_text(size = 2.5, vjust = -0.9, show.legend = FALSE) +
       labs(x = sprintf("PC1 (%.1f%%)", 100 * ve$var_explained[1]),
            y = sprintf("PC2 (%.1f%%)", 100 * ve$var_explained[2]),
            title = "Genotype PCA") + th, "pca.png", 6, 5)

# Windowed Fst and diversity ------------------------------------------------------
w <- rd("windows.tsv")
w$mid <- (w$start + w$end) / 2 / 1000
save(ggplot(w, aes(mid, fst_hudson)) + geom_line(colour = "grey50") + geom_point(size = 1.4, colour = "#b2182b") +
       facet_grid(. ~ chrom, scales = "free_x", space = "free_x") +
       labs(x = "Position (kb)", y = "Hudson Fst (window)", title = "Differentiation between groups") + th,
     "fst_windows.png", 9, 3.5)
pi_cols <- grep("^pi_", names(w), value = TRUE)
pil <- do.call(rbind, lapply(pi_cols, function(cl) data.frame(chrom = w$chrom, mid = w$mid,
                                                              group = sub("^pi_", "", cl), pi = w[[cl]])))
save(ggplot(pil, aes(mid, pi, colour = group)) + geom_line() +
       facet_grid(. ~ chrom, scales = "free_x", space = "free_x") +
       labs(x = "Position (kb)", y = expression(pi~"per bp"), title = "Nucleotide diversity") + th,
     "diversity_windows.png", 9, 3.5)

# Association scan ------------------------------------------------------------------
s <- rd("snp_sites.tsv")
if ("pvalue" %in% names(s)) {
  s$sig <- s$padj < 0.05
  save(ggplot(s, aes(pos / 1000, -log10(pvalue), colour = sig)) + geom_point(size = 0.9) +
         scale_colour_manual(values = c(`FALSE` = "grey60", `TRUE` = "#2166ac"), name = "FDR < 5%") +
         facet_grid(. ~ chrom, scales = "free_x", space = "free_x") +
         labs(x = "Position (kb)", y = expression(-log[10]~p), title = "SNP-group association (Fisher)") + th,
       "association_manhattan.png", 9, 3.5)
}

# Distance-based tree ------------------------------------------------------------------
D <- as.matrix(utils::read.delim(file.path(res, "distance_matrix.tsv"), row.names = 1, check.names = FALSE))
png(file.path(fig, "upgma_tree.png"), width = 7, height = 4.5, units = "in", res = 300)
plot(stats::hclust(stats::as.dist(D), method = "average"), main = "UPGMA of pairwise genetic distance",
     xlab = "", sub = "", ylab = "p-distance")
invisible(dev.off())

# SVs ---------------------------------------------------------------------------------
sv <- rd("sv_calls.tsv")
if (nrow(sv)) {
  save(ggplot(sv, aes(svlen, fill = svtype)) + geom_histogram(bins = 25, colour = "white") +
         scale_x_log10() + labs(x = "SV length (bp, log scale)", y = "Count", title = "Structural variants") + th,
       "sv_sizes.png", 6, 4)
}

# Resistance profile heatmap -------------------------------------------------------------
if (file.exists(file.path(res, "isolate_resistance_profile.tsv"))) {
  rp <- rd("isolate_resistance_profile.tsv")
  muts <- setdiff(names(rp), c("sample", "group"))
  long <- do.call(rbind, lapply(muts, function(m) data.frame(sample = rp$sample, group = rp$group,
                                                             mutation = m, carrier = factor(rp[[m]], 0:1, c("wild type", "mutant")))))
  long$sample <- factor(long$sample, levels = rp$sample[order(rp$group, rp$sample)])
  save(ggplot(long, aes(sample, mutation, fill = carrier)) + geom_tile(colour = "white") +
         scale_fill_manual(values = c(`wild type` = "grey85", mutant = "#b2182b"), name = NULL) +
         facet_grid(. ~ group, scales = "free_x", space = "free_x") +
         labs(x = NULL, y = NULL, title = "Target-site mutation profile per isolate") + th +
         theme(axis.text.x = element_text(angle = 90, vjust = 0.5, size = 7)),
       "resistance_profile.png", 9, 3)
}
message("Figures written to ", fig)
