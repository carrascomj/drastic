# Functions to build datasets to evaluate the performance of some trained model.
# Different datasets should be implemented:
#    1. Take random chunks of some size from the genome and label them.
#    2. Take overlapping windows from the input genome starting at the first
#           position.

import sys

sys.path.append("../pre-processing/")

from preproc_pipeline import window_pipeline


def build_chunks(
    genomes, feat_files, labels=["gene", "intergenic"], each=100, **kwargs
):
    """ Starting from a `genome` FASTA file and a `feat_file` feature table,
    builds a pandas dataframe of sparsed labeled sequences of size `window_size`.
    
    TODO: keep information about the genes to then analyse the results.
    
    Parameters
    ----------
    genomes: string || list of strings
        path to FASTA file(s)
    feat_files: string || list of strings
        path to corresponding(s) NCBI TSV feature file(s). Correspondence is encoded
        by position.
        TODO: specify how to get this file from NCBI
    labels: list of strings
        labels to include in the dataset
    **kwargs: named arguments
        other arguments to pass to window_pipeline. `window_size` is of extreme
        importance here. Not labels.

    Returns
    -------
    df_out: pandas.DataFrame
        each row is a sequence of size `window_size`. with `labels` and `features`
    """
    # stardadize input
    each = int(each)
    if type(genomes) is str:
        genomes = [genomes]
    if type(feat_files) is str:
        feat_files = [feat_files]

    df_out = pd.concat(
        [
            window_pipeline(genome, feat_file, **kwargs).sample(n=each)
            for genome, feat_file in zip(genomes, feat_files)
        ]
    )
    labels_in_df = np.unique(df_out.label)
    if set(labels_in_df) != set(labels):
        # sequences for certain labels won't be used
        df_out = df_out[df.label.isin(labels)]

    return df_out


def slide_genome():
    raise NotImplementedError("Further in the horizon!")
