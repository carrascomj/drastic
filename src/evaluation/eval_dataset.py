"""
Functions to build datasets to evaluate the performance of some trained model.

Different datasets should be implemented:
    1. Take random chunks of some size from the genome and label them.
    2. Take overlapping windows from the input genome starting at the first
       position.
"""

import sys

sys.path.append("../pre-processing/")

# from preproc_pipeline import window_pipeline
from torch.utils.data import DataLoader, Dataset
from warnings import simplefilter


class oversampdata(Dataset):
    def __init__(self, data):
        # first column is list of index sentence
        self.data = torch.LongTensor(
            np.ndarray.astype(
                np.array([np.array(r) for r in data.iloc[:, 0].to_numpy()]), "int64"
            )
        )
        # second column is the label
        self.targets = torch.LongTensor(
            np.ndarray.astype(
                np.array([np.array(r) for r in data.iloc[:, 1].to_numpy()]), "int64"
            )
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_val = self.data[index]
        target = self.targets[index]
        return data_val, target


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


def retrieve_annotations(fseq_idx, feature_table, sep="\t"):
    annotations = pd.read_csv(feature_table, sep)
    true_annotations = []

    for index, row in annotations.iterrows():
        if row["start"] in fseq_idx:
            true_annotations.append(row["start"])
        if row["end"] in fseq_idx:
            true_annotations.append(row["end"])
    return true_annotations


def annotate(idx_window, true_annot):
    for idx, annot in enumerate(true_annot):
        if annot in idx_window:
            # Start gene
            if idx % 2 == 0:
                if idx_window[-1] - annot >= 0.9 * len(idx_window):
                    return 1
            # End gene
            else:
                if annot - idx_window[0] >= 0.9 * len(idx_window):
                    return 1

        # Within genes
        if idx_window[0] > annot and idx_window[-1] < true_annot[idx + 1]:
            return 1
    return 0


def cut_annotate_seq(fseq, fseq_idx, true_annot, window_size, slide):
    seq = []
    seq_annot = []
    N = len(fseq)
    for i in range(0, N - window_size + 1, slide):
        seq.append(fseq[i : i + window_size])

        idx_window = fseq_idx[i : i + window_size]
        seq_annotation = annotate(idx_window, true_annot)
        seq_annot.append(seq_annotation)

    return (seq, seq_annot)


def slide_genome(
    genomes, labels, feat_files, indeces, window_size, slide=5, sep="\t", **kwargs
):
    """Take overlapping windows of size `window_size` of the genomes
    
    Parameters
    ----------
    genomes: list
        list of paths (str) to genome files
    feat_files: list
        list of paths (str) to feature files
    indeces: list
        list of lists (len 2), indication the positions that correspond to each
        evaluation chunk in the genome
    window_size: int
        size of chunks
    slide: int
        every step of the windows
    sep: "\t"
        separator character of the feat_file table
    **kwargs:
        Ignored. Just for consistency with other post-processing methods.

    Result
    ------
    testing_loader torch.utils.data.DataLoader
        evaluation object

    """
    simplefilter("ignore")
    # for the one hot encoding of lavbels
    lab2vec = {k: i for i, k in enumerate(labels)}
    # put all the genome eval info on one big dataframe
    # the information of the evaluation comes from half of the fasta file of each
    # genomes (some of them are the first half, some of them, the second)
    # the dataset used for training is the opposite part
    dfs = []
    for genome_file, feature_table, index in zip(genomes, feat_files, indeces):
        fna = pyfastx.Fasta(genome_file)
        genome = str(fna[0])
        fqindex = range(index[0], index[1])

        true_annotations = retrieve_annotations(fqindex, genome, sep)
        (processed_test_seq, test_annotation) = cut_annotate_seq(
            genome, fqindex, true_annotations, window_size, slide
        )
        df_testing = pd.DataFrame(dict(sequence=processed_test_seq))
        df_testing["c_class"] = df_testing["sequence"].apply(
            lambda x: code_one_hot(x, nuc_to_class)
        )
        df_testing["Annotation"] = test_annotation

        num_to_onehot = {0: [0, 1], 1: [1, 0]}
        df_testing["onehot_label"] = df_testing["Annotation"].apply(
            lambda x: class2onehot(x, num_to_onehot)
        )
        dfs.append(df)
    dfs = pd.concat(dfs)

    testing_dataset = oversampdata(dfs.loc[:, ["c_class", "onehot_label"]])
    testing_loader = torch.utils.data.DataLoader(
        testing_dataset, batch_size=1, shuffle=False, drop_last=False
    )
    return testing_loader
