import re, argparse
from collections import defaultdict
import pathlib as pl
import pandas as pd


def get_f1(file_content: str, header: str) -> float:
    m = re.search(
        f"{header}\n"
        r"Correct:[^\n]+\n"
        r"Spurious:[^\n]+\n"
        r"Precision:[^\n]+\n"
        r"F1: ([0-9.]+)",
        file_content,
    )
    if m is None:
        raise ValueError("Cannot find type F1")
    return float(m.groups()[0])


def get_type_f1(file_content: str) -> float:
    return get_f1(file_content, "Ent_type")


def get_partial_f1(file_content: str) -> float:
    return get_f1(file_content, "Partial")


def get_strict_f1(file_content: str) -> float:
    return get_f1(file_content, "Strict")


def get_exact_f1(file_content: str) -> float:
    return get_f1(file_content, "Exact")


def format_dataset_name(name: str) -> str:
    dataset_components = re.split(r"[:_]", name)
    dataset_name = dataset_components[0]
    if any(c == "multi" for c in dataset_components):
        dataset_name += "multi"
    if any(m := re.match(r"retimestamped-([0-9]+)", c) for c in dataset_components):
        assert not m is None
        dataset_name = "{} => {}".format(dataset_name, m.group(1))
    return dataset_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", type=pl.Path)
    args = parser.parse_args()

    df_dict = defaultdict(list)
    for score_file in args.directory.glob("*_score.txt"):
        dataset = format_dataset_name(score_file.name)
        df_dict["dataset"].append(dataset)
        with open(score_file) as f:
            content = f.read()
        df_dict["type"].append(get_type_f1(content))
        df_dict["partial"].append(get_partial_f1(content))
        df_dict["strict"].append(get_strict_f1(content))
        df_dict["exact"].append(get_exact_f1(content))

    df = pd.DataFrame(df_dict)
    f1_cols = ["type", "partial", "strict", "exact"]
    df[f1_cols] = df[f1_cols].apply(lambda v: round(v * 100, 2))
    print(df)
