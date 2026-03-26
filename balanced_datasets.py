import argparse, shutil, ast, random, re
from datetime import datetime
from collections import Counter, defaultdict
import pathlib as pl

Fact = list[str]


def mm(date: str) -> str:
    date_dt = datetime.strptime(date, "%Y-%m-%d")
    return date_dt.strftime("%m")


def downsample_rel(
    fact_descs: list[str],
    ref_list: list[list[Fact]],
    rel_limit: dict[str, int],
) -> tuple[list[str], list[list[Fact]]]:
    """Downsample a dataset in terms of relations

    :param fact_descs: list[description]
    :param ref_list: list[list[quad] <- one list[quad] per line]
    :param rel_limit: a dictionary mapping a relation to a
        maximum number of facts with that relation to keep

    :return: a tuple with a downsampled list of fact descriptions and
             a downsampled list of references.
    """
    rel_counter = defaultdict(int)
    downsampled_fact_descs = []
    downsampled_ref_list = []

    for fact, quads in zip(fact_descs, ref_list):
        fact_rel_counter = Counter([rel for _, rel, _, _ in quads])
        if any(
            rel_counter[rel] + count > rel_limit[rel]
            for rel, count in fact_rel_counter.items()
        ):
            continue
        downsampled_fact_descs.append(fact)
        downsampled_ref_list.append(quads)
        for rel, count in fact_rel_counter.items():
            rel_counter[rel] += count

    return (downsampled_fact_descs, downsampled_ref_list)


def counter_diff(counter1: Counter, counter2: Counter) -> float:
    diff = 0
    keys = set(counter1.keys()).union(counter2.keys())
    for k in keys:
        v1 = counter1.get(k, 0)
        v2 = counter2.get(k, 0)
        diff += abs(v1 - v2)
    return diff / max(sum(counter1.values()), sum(counter2.values()))


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
    rel_counter_1 = Counter([rel for quads in ref_list1 for _, rel, _, _ in quads])
    rel_counter_2 = Counter([rel for quads in ref_list2 for _, rel, _, _ in quads])
    # in the case of multi facts, downsampling per relation might not
    # be sufficient. However, repeating the process can succeed.
    max_tries = 100
    tries_nb = 0
    while counter_diff(rel_counter_1, rel_counter_2) > 0.05 and tries_nb < max_tries:
        all_rels = set(rel_counter_1.keys()).union(set(rel_counter_2.keys()))
        rel_limit = {
            rel: min(rel_counter_1.get(rel, 0), rel_counter_2.get(rel, 0))
            for rel in all_rels
        }

        ts_counter_1 = Counter([mm(ts) for quads in ref_list1 for _, _, _, ts in quads])
        ts_counter_2 = Counter([mm(ts) for quads in ref_list2 for _, _, _, ts in quads])
        all_ts = set(ts_counter_1.keys()).union(set(ts_counter_2.keys()))
        ts_limit = {
            ts: min(ts_counter_1.get(ts, 0), ts_counter_2.get(ts, 0)) for ts in all_ts
        }

        fact_descs1, ref_list1 = downsample_rel(fact_descs1, ref_list1, rel_limit)
        fact_descs2, ref_list2 = downsample_rel(fact_descs2, ref_list2, rel_limit)

        rel_counter_1 = Counter([rel for quads in ref_list1 for _, rel, _, _ in quads])
        rel_counter_2 = Counter([rel for quads in ref_list2 for _, rel, _, _ in quads])
        tries_nb += 1

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

    new_ref_list1, new_fact_descs1 = [], []
    new_ref_list2, new_fact_descs2 = [], []
    for month in [
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "08",
        "09",
        "10",
        "11",
        "12",
    ]:
        month1 = [
            (ref, desc)
            for ref, desc in zip(ref_list1, fact_descs1)
            if any(mm(fact[3]) == month for fact in ref)
        ]
        month2 = [
            (ref, desc)
            for ref, desc in zip(ref_list2, fact_descs2)
            if any(mm(fact[3]) == month for fact in ref)
        ]
        month_fact_descs1, month_ref_list1, month_fact_descs2, month_ref_list2 = (
            balance(
                [desc for _, desc in month1],
                [ref for ref, _ in month1],
                [desc for _, desc in month2],
                [ref for ref, _ in month2],
            )
        )
        new_ref_list1 += month_ref_list1
        new_fact_descs1 += month_fact_descs1
        new_ref_list2 += month_ref_list2
        new_fact_descs2 += month_fact_descs2

    write_balanced_dataset(
        args.first_dataset, new_fact_descs1, new_ref_list1, args.second_dataset
    )
    write_balanced_dataset(
        args.second_dataset, new_fact_descs2, new_ref_list2, args.first_dataset
    )
