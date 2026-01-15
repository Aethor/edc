import argparse, shutil, ast, random, re
from datetime import datetime
from collections import Counter, defaultdict
import pathlib as pl

Fact = list[str]


def mm_dd(date: str) -> str:
    date_dt = datetime.strptime(date, "%Y-%m-%d")
    return date_dt.strftime("%m-%d")


def downsample(
    fact_descs: list[str],
    ref_list: list[list[Fact]],
    rel_limit: dict[frozenset[str], int],
    ts_limit: dict[frozenset[str], int],
) -> tuple[list[str], list[list[Fact]]]:
    """Greedily downsample a dataset in terms of relations, according
    to rel_limit and ts_limit.

    :param fact_descs: list[description]
    :param ref_list: list[list[quad] <- one list[quad] per line]
    :param rel_limit: a dictionary mapping a set of relations to a
        maximum number of examples to keep with that set of relations
    :param ts_limit: a dictionary mapping a set of MM-DD timestamps to
        a maximum number of examples to keep with that set of
        timestamps

    :return: a tuple with a downsampled list of fact descriptions and
             a downsampled list of references.
    """
    rel_counter = defaultdict(int)
    ts_counter = defaultdict(int)
    downsampled_fact_descs = []
    downampled_ref_list = []

    for fact, quads in zip(fact_descs, ref_list):
        rel_set = frozenset(rel for _, rel, _, _ in quads)
        ts_set = frozenset(mm_dd(ts) for _, _, _, ts in quads)
        if (
            rel_counter[rel_set] >= rel_limit[rel_set]
            or ts_counter[ts_set] >= ts_limit[ts_set]
        ):
            continue
        downsampled_fact_descs.append(fact)
        downampled_ref_list.append(quads)
        rel_counter[rel_set] += 1
        ts_counter[ts_set] += 1

    return (downsampled_fact_descs, downampled_ref_list)


def balance(
    fact_descs1: list[str],
    ref_list1: list[list[Fact]],
    fact_descs2: list[str],
    ref_list2: list[list[Fact]],
) -> tuple[list[str], list[list[Fact]], list[str], list[list[Fact]]]:
    """Balance two datasets in terms of relations and MM-DD timestamps
    by downsampling

    :param fact_descs1: list[description]
    :param ref_list1: list[list[quad] <- one list[quad] per line]
    :param fact_descs2: list[description]
    :param ref_list2: list[list[quad] <- one list[quad] per line]

    :return: (fact_descs1, ref_list1, fact_descs2, ref_list2)
    """
    rel_counter_1 = Counter(
        [frozenset(rel for _, rel, _, _ in quads) for quads in ref_list1]
    )
    rel_counter_2 = Counter(
        [frozenset(rel for _, rel, _, _ in quads) for quads in ref_list2]
    )
    all_rel_set = set(rel_counter_1.keys()).union(set(rel_counter_2.keys()))
    rel_limit = {
        rel_set: min(rel_counter_1.get(rel_set, 0), rel_counter_2.get(rel_set, 0))
        for rel_set in all_rel_set
    }

    ts_counter_1 = Counter(
        [frozenset(mm_dd(ts) for _, _, _, ts in quads) for quads in ref_list1]
    )
    ts_counter_2 = Counter(
        [frozenset(mm_dd(ts) for _, _, _, ts in quads) for quads in ref_list2]
    )
    all_ts_set = set(ts_counter_1.keys()).union(set(ts_counter_2.keys()))
    ts_limit = {
        ts_set: min(ts_counter_1.get(ts_set, 0), ts_counter_2.get(ts_set, 0))
        for ts_set in all_ts_set
    }

    fact_descs1, ref_list1 = downsample(fact_descs1, ref_list1, rel_limit, ts_limit)
    fact_descs2, ref_list2 = downsample(fact_descs2, ref_list2, rel_limit, ts_limit)

    return (fact_descs1, ref_list1, fact_descs2, ref_list2)


def escape_single_quotes(elt: str) -> str:
    return re.sub(r"'", "\\'", elt)


def escape_quad_single_quotes(quad: Fact) -> Fact:
    subj, rel, obj, ts = quad
    return [
        escape_single_quotes(subj),
        escape_single_quotes(rel),
        escape_single_quotes(obj),
        ts,
    ]


def write_balanced_dataset(
    name: str, fact_descs: list[str], ref_list: list[list[Fact]], twin_name: str
):
    balanced_name = f"{name}:balanced-{twin_name}"

    # fact descriptions
    out_fact_descs_path = pl.Path("./dsets") / f"{balanced_name}.txt"
    print(f"writing {out_fact_descs_path}...", end="")
    with open(out_fact_descs_path, "w") as f:
        f.writelines(fact_descs)
    print("done!")

    # references
    refs_path = pl.Path("./evaluate/references") / f"{balanced_name}.txt"
    print(f"writing {refs_path}...", end="")
    with open(refs_path, "w") as f:
        for ref in ref_list:
            f.write("[")
            quad_strings = []
            for quad in ref:
                s, r, o, t = escape_quad_single_quotes(quad)
                quad_strings.append(f"['{s}', '{r}', '{o}', '{t}']")
            f.write(", ".join(quad_strings))
            f.write("]\n")
    print("done!")

    # few-shot examples
    in_few_shot_examples_dir = pl.Path("./few_shot_examples") / name
    out_few_shot_examples_dir = pl.Path("./few_shot_examples") / balanced_name
    print(
        f"copying {in_few_shot_examples_dir} to {out_few_shot_examples_dir} directory...",
        end="",
    )
    shutil.copytree(
        in_few_shot_examples_dir, out_few_shot_examples_dir, dirs_exist_ok=True
    )
    print("done!")

    # schema
    in_schema = pl.Path("./schemas") / f"{name}_schema.csv"
    out_schema = pl.Path("./schemas") / f"{balanced_name}_schema.csv"
    print(f"copying schema file {in_schema} to {out_schema}...", end="")
    shutil.copy(in_schema, out_schema)
    print("done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d1",
        "--first-dataset",
        type=str,
        help="the first dataset on which to evaluate",
    )
    parser.add_argument(
        "-d2",
        "--second-dataset",
        type=str,
        help="the second dataset on which to evaluate",
    )
    args = parser.parse_args()

    random.seed(0)

    with open(f"./dsets/{args.first_dataset}.txt") as f:
        fact_descs1 = f.readlines()
    with open(f"./evaluate/references/{args.first_dataset}.txt") as f:
        ref_list1 = [ast.literal_eval(quads.strip()) for quads in f.readlines()]

    with open(f"./dsets/{args.second_dataset}.txt", "r") as f:
        fact_descs2 = f.readlines()
    with open(f"./evaluate/references/{args.second_dataset}.txt") as f:
        ref_list2 = [ast.literal_eval(quads) for quads in f.readlines()]

    fact_descs1, ref_list1, fact_descs2, ref_list2 = balance(
        fact_descs1, ref_list1, fact_descs2, ref_list2
    )
    write_balanced_dataset(
        args.first_dataset, fact_descs1, ref_list1, args.second_dataset
    )
    write_balanced_dataset(
        args.second_dataset, fact_descs2, ref_list2, args.first_dataset
    )
