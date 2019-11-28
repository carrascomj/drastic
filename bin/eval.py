# CLI to evaluate a model

import sys

sys.path.append("../src/evaluation")
sys.path.append("../src/models")

from eval_dataset import build_chunks, slide_genome
from analysis import evaluate, to_confussion_matrix

# Default arguments
expected_args = {
    "--model": None,
    "--genomes": [
        "0_0_TEST_GCF_000008525.1_ASM852v1_genomic.fna",
        "1_0_TEST_GCF_000008665.1_ASM866v1_genomic.fna",
        "2_0_TEST_GCF_000008865.2_ASM886v2_genomic.fna",
        "3_0_TEST_GCF_000009045.1_ASM904v1_genomic.fna",
        "4_0_TEST_GCF_000027305.1_ASM2730v1_genomic.fna",
        "5_0_TEST_GCF_000027325.1_ASM2732v1_genomic.fna",
        "6_0_TEST_GCF_000027345.1_ASM2734v1_genomic.fna",
        "7_0_TEST_GCF_000091665.1_ASM9166v1_genomic.fna",
        "8_0_TEST_GCF_000214725.1_ASM21472v1_genomic.fna",
        "9_0_TEST_GCF_003015225.1_ASM301522v1_genomic.fna",
    ],
    "--feat_files": [
        "GCA_000008525.1_ASM852v1_feature_table.tsv",
        "GCA_000008665.1_ASM866v1_feature_table.tsv",
        "GCA_000008865.2_ASM886v2_feature_table.tsv",
        "GCA_000009045.1_ASM904v1_feature_table.tsv",
        "GCA_000027305.1_ASM2730v1_feature_table.tsv",
        "GCA_000027325.1_ASM2732v1_feature_table.tsv",
        "GCA_000027345.1_ASM2734v1_feature_table.tsv",
        "GCA_000091665.1_ASM9166v1_feature_table.tsv",
        "GCA_000214725.1_ASM21472v1_feature_table.tsv",
        "GCA_003015225.1_ASM301522v1_feature_table.tsv",
    ],
    "--labels": ["intergenic", "gene"],
    "--each": 100,
    "--window_size": 50,
    "--num_windows": 3,
    "--min_gene_size": 200,
    "--max_gene_size": 2000,
    "--sep": "\t",
    "--out_file": False,
    "--method": "slide_genome",
}


def parse_list_args(arg):
    """ Parse a string list of the form --some_arg=elem1,elem2,elem3 """
    eq_where = arg.find("=")
    if eq_where == -1:
        eq_where = 0
    else:
        return arg[eq_where + 1 :].split(",")


def startswith_dict(key, d):
    """ Check if the string `key` starts any key in the dict `d` """
    for k in d:
        if k.startswith(key):
            return True
    else:
        return False


def loop_args():
    global expected_args
    args = sys.argv[1:]
    for arg in args:
        pure = arg[0 : arg.find("=")]
        if startswith_dict(pure, expected_args):
            expected_args[pure] = parse_list_args(arg)
            if len(expected_args[pure]) == 0:
                expected_args[pure] = None
            elif len(expected_args[pure]) == 1:
                expected_args[pure] = expected_args[pure][0]
        else:
            print(f"Parameter {arg} not recognized, ignoring that parameter")
    return expected_args


def load_model(model_class):
    """Instantiate the `model_class` object and load its trained weights."""
    if model_class.lower() == "convlstm":
        from conv_LSTM import convLSTM, criterion

        model = convLSTM()
        path = "../data/models/conv_lstm.pt"
        model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))

    else:
        raise NotImplementedError(
            "Need to set another step from sequences to indexes using wti.p!"
        )
        path = "../data/models/rnn_embed.pt"
        from embeddings_LSTM import convLSTM, criterion

        model = embedLSTM()

    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    crit = criterion()

    return model, crit


def apply_evaluation(args):
    """Gather all the functions in the evaluation workflow."""
    # model and method are treated differently (aren't arguments to build datasets)
    model = args["--model"]
    method = build_chunks if args["--model"] == "build_chuncks" else slide_genome

    # Evaluation dataset: get all argumetns without the starting "--"
    kwargs = {k[2:]: v for k, v in args.items() if k not in ["--model", "--method"]}
    df_in = method(**kwargs)
    # load the models and evaluate
    load_model(model)
    loss, accuracy, predictions_all, labels_all = evaluate(model, df_in, criterion)
    conf_matrix = confussion_matrix(predictions_all, labels_all)
    return loss, accuracy, conf_matrix


if __name__ == "__main__":
    evaluated = apply_evaluation(loop_args())
    print(evaluated)
