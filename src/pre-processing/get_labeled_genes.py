# Take all the genes of a genome, with varying filling space before and after them

import numpy as np
import pandas as pd

from numpy.random import randint, random, choice
from warnings import warn


def get_feature_ranges(feat_table, feature="gene"):
    """ Get all ranges that define genes/proteins (`feature`) in a genome

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


def get_negative_ranges(feat_ranges):
    """ Starting from a set of ranges that defines classified input, get the
    complementary ranges to sample negative observations

    Parameters
    ----------
    feat_ranges : numpy.array
        each position defines a gene/protein, a numpy array of length 2 (start and end)

    Returns
    -------
    numpy.array
        each position defines a negative observation, a numpy array of length 2 (start and end)
    """
    negative_ranges = []
    # edge case: the first gene starts at position 0
    if feat_ranges[0, 0] == 0:
        lb, ub = feat_ranges[0, 1] + 1, feat_ranges[1, 0] - 1
        start = 1
    else:
        lb, ub = 0, feat_ranges[0, 0] - 1
        start = 0
    for i in range(start, len(feat_ranges) - 1):
        # print(f"ub -> {ub}\tfeat_ranges[i,0] -> {feat_ranges[i,0]}\ti -> {i}")
        if ub < feat_ranges[i, 0]:
            # just update lower bound and add range if genes don't overlap
            negative_ranges.append(np.array([lb, ub]))
            lb = feat_ranges[i, 1] + 1
        ub = feat_ranges[i + 1, 0] - 1
    return np.array(negative_ranges)


def _bootstrap(input_arr):
    """ Take *N* samples with replacement of `input_arr`, where *N* is the initial
    length of `input_arr`.

    Parameters
    ----------
    input_arr : numpy.array

    Returns
    -------
    numpy.array

    """
    n = len(input_arr)
    return input_arr[choice(n, size=n, replace=True)]


def _sample_in_range(feat_range, min_size, max_size):
    """ Take a subset of a range defined in `feat_range` """
    size = feat_range[1] - feat_range[0]

    if min_size >= size:
        warn(
            "min_size parameter ("
            + str(min_size)
            + ") is bigger or equal to the size of"
            "input range" + str(feat_range) + ". Ignoring that range..."
        )
        return feat_range

    left = randint(feat_range[0], feat_range[1] - min_size)
    right = randint(left + min_size, min(left + max_size, feat_range[1]))
    return np.array([left, right])


def sample_negatives(feat_ranges, min_size=50, max_size=1000):
    """ Take negative example ranges of some defined length """
    sampled = np.array(list(filter(lambda x: x[1] - x[0] >= min_size, feat_ranges)))
    sampled = [_sample_in_range(r, min_size, max_size) for r in _bootstrap(sampled)]
    # sort just to check output in tests
    return np.array(sorted(sampled, key=lambda x: x[0]))


def _tweak_range(feat_range, min_fill, max_fill, gene_percentage, p):
    """ Expand a range """
    new_range = np.zeros(2, dtype=np.int32)
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
        tweaked[-1, -1] = genome_length
    return tweaked


if __name__ == "__main__":
    df_test = pd.read_csv(
        "../../data/GCA_000008525.1_ASM852v1_feature_table.tsv", sep="\t"
    )
    gene_ranges = get_feature_ranges(df_test)
    negative_ranges = get_negative_ranges(gene_ranges)
    sampled_negatives = sample_negatives(negative_ranges)
    tweaked_gene_ranges = tweak_ranges(gene_ranges, genome_length=60000)
    for i in range(5):
        print(
            f"Actual gene -> {gene_ranges[i]}\nTweaked gene -> {tweaked_gene_ranges[i]}\n"
            f"Negative observation -> {negative_ranges[i]}\n"
            f"Sampled negative -> {sampled_negatives[i]}\n"
        )