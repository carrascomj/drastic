# CLI to evaluate a model

import sys

sys.path.append("../src/evaluation")

from eval_dataset import build_chunks, slide_genome
from analysis import loop_eval, confussion_matrix

expected_args = {
    "--model": None,
    "--genomes": None,
    "--feat_files": None,
    "--labels": ["intergenic", "gene"],
    "--each": 100,
    "--window_size": 50,
    "--num_windows": 3,
    "--min_gene_size": 200,
    "--max_gene_size": 2000,
    "--sep": "\t",
    "--out_file": False,
    "--method": "build_chunks",
}


def parse_list_args(arg):
    """ Parse a string list of the form --some_arg=elem1,elem2,elem3 """
    eq_where = arg.find("=")
    if eq_where == -1:
        eq_where = 0
    else:
        return arg[eq_where:].split(",")


def startswith_dict(key, d):
    """ Check if the string `key` starts any key in the dict `d` """
    for k in d:
        if k.startswith(key):
            return True
    else:
        return False


def loop_args():
    args = sys.argv[1:]
    for arg in args:
        if startswith_dict(arg, expected_args):
            expected_args[arg] = parse_list_args(arg)
            if len(expected_args[arg]) == 0:
                expected_args[arg] = expected_args[arg][0]
        else:
            print(f"Parameter {arg} not recognized, ignoring that parameter")
    return expected_args


def apply_evaluation(args):
    # model and method are treated differently (aren't arguments to build datasets)
    model = args["--model"]
    method = build_chunks if args["--model"] == "build_chuncks" else slide_genome

    # get all argumetns without the starting "--"
    kwargs = {k[2:]: v for k, v in args.items() if k not in ["--model", "--method"]}

    df_in = method(**kwargs)
    df_out = loop_eval(model, df_out)
    conf_matrix = confussion_matrix(df_out)
    return conf_matrix


if __name__ == "__main__":
    print(apply_evaluation(loop_args()))
