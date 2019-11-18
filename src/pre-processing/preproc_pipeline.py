# High-level functions that join all the pre-processing routines into a pipeline

import pandas as pd
import pyfastx

from build_dataset import build_padded_seqs, map_full_genomes
from get_labeled_genes import (
    get_feature_ranges,
    tweak_ranges,
    get_negative_ranges,
    sample_negatives,
    get_partial_ranges,
    extract_windows,
)
from remove_small_genes import remove_small_genes


def pre_process(
    genome_file,
    feat_file,
    min_gene_size=200,
    max_gene_size=2000,
    sep="\t",
    out_file=False,
):
    """ Master function
    
    Accepts a `genome_file` FASTA and a `feat_file` TSV and convers it to a pandas
    dataframe.
    
    Parameters
    ----------
    genome_file: string
        path to FASTA file
    feat_file: string
        path to corresponding NCBI TSV feature file 
        TODO: specify how to get this file from NCBI
    min_gene_size: int
        minimum gene size accepted as input. Defautl to 200
    sep: string
        sep argument to the pandas's read and write functions. Default to "\t"
    out_file: False || string
        if a string is passed, write the dataframe to a file `string`. Default to False
        
    Returns
    -------
    pandas.DataFrame
        final dataframe to use in training/testing
    """
    # 0. CONSISTENCY OF INPUT
    min_gene_size = int(min_gene_size)

    # 1. READ FILES
    # We expect to have the whole genome as the fist sequence in the multiFASTA
    # file and then some plasmids.
    # TODO: check if that works for a 1-seq FASTA file
    fna = pyfastx.Fasta(genome_file)
    genome = str(fna[0])
    del fna
    df_feat = pd.read_csv(feat_file, sep=sep)
    genome_length = len(genome)

    # 2. STANDARDIZE DATA
    # TODO: remove also big genes
    df_feat = remove_small_genes(df_feat, min_gene_size)

    # 3. GET RANGES OF FEATURES
    # Test genes
    gene_ranges = get_feature_ranges(df_feat)
    positive = tweak_ranges(gene_ranges, genome_length=genome_length)
    # Test negative examples
    negative_ranges = get_negative_ranges(gene_ranges)
    negative = sample_negatives(negative_ranges)
    # Test partial genes
    partial = get_partial_ranges(gene_ranges, genome_length=genome_length)

    # 4. BUILD THE PRE-PROCESSED DATAFRAME
    df_out = map_full_genomes(genome, positive, negative, partial)
    df_out = df_out[df_out.sequence.apply(lambda x: len(x)) < max_gene_size]
    df_out = build_padded_seqs(df_out)

    if out_file:
        df_out.to_csv(out_file, sep=sep, index=False)
    return df_out


def window_pipeline(
    genome_file,
    feat_file,
    window_size=50,
    num_windows=3,
    min_gene_size=200,
    max_gene_size=2000,
    sep="\t",
    out_file=False,
):
    """ Preprocessing pipelines that take equidistant windows in the features ranges """
    # 0. CONSISTENCY OF INPUT
    window_size = int(window_size)
    num_windows = int(num_windows)

    # 1. READ FILES
    # We expect to have the whole genome as the fist sequence in the multiFASTA
    # file and then some plasmids.
    # TODO: check if that works for a 1-seq FASTA file
    fna = pyfastx.Fasta(genome_file)
    genome = str(fna[0])
    # just avoid segmentation problems with the lower level API
    del fna
    df_feat = pd.read_csv(feat_file, sep=sep)
    genome_length = len(genome)

    # 2. STANDARIZE DATA
    df_feat = remove_small_genes(df_feat, min_gene_size)

    # 3. GET RANGES OF FEATURES
    # Test genes
    gene_ranges = get_feature_ranges(df_feat)
    positive = extract_windows(gene_ranges, window_size, num_windows)
    # Test negative examples
    negative_ranges = get_negative_ranges(gene_ranges)
    negative = extract_windows(negative_ranges, window_size, num_windows)
    # Test partial genes
    partial_ranges = get_partial_ranges(gene_ranges, genome_length=genome_length)
    partial = extract_windows(partial_ranges, window_size, num_windows)

    # 4. BUILD THE PRE-PROCESSED DATAFRAME
    df_out = map_full_genomes(genome, positive, negative, partial)

    return df_out


if __name__ == "__main__":
    # Test with Escherichia coli's genome and feature table
    genome = "../../data/GCF_000008865.2_ASM886v2_genomic.fna"
    feature_table = "../../data/GCA_000008865.2_ASM886v2_feature_table.tsv"
    df = window_pipeline(genome, feature_table)
    print(
        f"\nRESULT OF TEST\n{len('RESULT OF TEST')*'='}\n{df.head(20)}\n"
        f"colums -> {list(df.columns)}"
    )
    assert not df.sequence.apply(lambda x: len(x) != 50).any()
