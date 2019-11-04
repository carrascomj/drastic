# Take all the genes of a genome, with varying filling space before and after them

import numpy as np
import pandas as pd

from numpy.random import randint, random


def get_feature_ranges(feat_table, feature="gene"):
    """ Get all ranges that defines genes/proteins (`feature`) in a genome

    Parameters
    ----------
    feat_table : pandas.DataFrame
        a feature table from a genome
    feature: str
        "gene" || "protein"

    Returns
    ------
    numpy.array
        each position defines a gene/protein, a numpy array of length 2 (start and end)
    """
    return (
        feat_table[feat_table["# feature"] == feature]
        .loc[:, ["start", "end"]]
        .to_numpy()
    )


def _tweak_range(feat_range, min_fill, max_fill, gene_percentage, p):
    """
    Expand a range
    """
    new_range = np.zeros(2)
    size = feat_range[1] - feat_range[0]
    expand = lambda: randint(min_fill, max_fill)
    chop = lambda: randint(0, (1 - gene_percentage) * size)
    # left extreme
    if random() < p:
        new_range[0] = feat_range[0] - expand()
    else:
        chopped = chop()
        new_range[0] = feat_range[0] + chopped
        size -= chopped
    # right extreme
    if random() < p:
        new_range[1] = feat_range[1] + expand()
    else:
        new_range[1] = feat_range[1] - chop()
    return new_range


def tweak_ranges(
    feat_ranges, genome_length, min_fill=5, max_fill=200, gene_percentage=0.9, p=0.9
):
    """ Expand each range that defines a gene. A sequence is labeled as "gene" if
    it contains the 90% of the gene, so some genes are randomly sequeezed.

    Parameters
    ----------
    feat_ranges : numpy.array
        each position defines a gene/protein, a numpy array of length 2 (start and end)
    p: float
        probability of a gene side being expanded. 1-p is the probability for the
        gene being squezeed.

    Returns
    ------
    numpy.array
        each position defines a gene/protein, a numpy array of length 2 (start and end)
    """
    tweaked = np.array(
        [
            _tweak_range(rang, min_fill, max_fill, gene_percentage, p)
            for rang in feat_ranges
        ]
    )
    # adjust edges to the genome
    if tweaked[0, 0] < 0:
        tweaked[0, 0] = 0
    if tweaked[-1, -1] > genome_length:
        tweaked[-1, -1] = 0
    return tweaked


if __name__ == "__main__":
    df_test = pd.read_csv(
        "../../data/GCA_000008525.1_ASM852v1_feature_table.tsv", sep="\t"
    )
    gene_ranges = get_feature_ranges(df_test)
    tweaked_gene_ranges = tweak_ranges(gene_ranges, genome_length=60000)
    for i in range(5):
        print(
            f"Actual gene -> {gene_ranges[i]}\nTweaked gene -> {tweaked_gene_ranges[i]}\n"
        )
