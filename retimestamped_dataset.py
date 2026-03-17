import argparse, shutil, re
import pathlib as pl


def replace_map(s: str, replacements: dict[str, str]) -> str:
    """
    For each key of replacements, replace each occurrence in s by
    replacements[key].

    >>> replace_map('2021 2022 2023', {'2021': '2020', '2022': '2021', '2023': '2022'})
    '2020 2021 2022'
    """
    p = "(" + "|".join(replacements.keys()) + ")"
    return re.sub(p, lambda m: replacements.get(m.group(0)), s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dataset", type=str)
    parser.add_argument("-o", "--output-dataset", type=str)
    parser.add_argument("-oy", "--old-year", type=str)
    parser.add_argument("-ny", "--new-year", type=str)
    args = parser.parse_args()
    old_year_prev = str(int(args.old_year) - 1)
    old_year_next = str(int(args.old_year) + 1)
    new_year_prev = str(int(args.new_year) - 1)
    new_year_next = str(int(args.new_year) + 1)
    rmap = {
        old_year_prev: new_year_prev,
        args.old_year: args.new_year,
        old_year_next: new_year_next,
    }

    with open(pl.Path("./dsets") / (args.input_dataset + ".txt")) as f:
        fact_descs = f.readlines()
    fact_descs = [replace_map(desc, rmap) for desc in fact_descs]

    with open(pl.Path("./evaluate/references") / (args.input_dataset + ".txt")) as f:
        refs = f.readlines()
    assert len(fact_descs) == len(refs)
    refs = [replace_map(r, rmap) for r in refs]

    out_fact_descs_path = pl.Path("./dsets") / (args.output_dataset + ".txt")
    print(f"writing {out_fact_descs_path}...", end="")
    with open(out_fact_descs_path, "w") as f:
        fact_descs[-1] = fact_descs[-1].rstrip("\n")
        f.writelines(fact_descs)
    print("done!")

    refs_path = pl.Path("./evaluate/references") / (args.output_dataset + ".txt")
    print(f"writing {refs_path}...", end="")
    with open(refs_path, "w") as f:
        refs[-1] = refs[-1].rstrip("\n")
        f.writelines(refs)
    print("done!")

    in_few_shot_examples_dir = pl.Path("./few_shot_examples") / args.input_dataset
    out_few_shot_examples_dir = pl.Path("./few_shot_examples") / args.output_dataset
    print(
        f"copying {in_few_shot_examples_dir} to {out_few_shot_examples_dir} directory...",
        end="",
    )
    shutil.copytree(
        in_few_shot_examples_dir, out_few_shot_examples_dir, dirs_exist_ok=True
    )
    print("done!")

    in_schema = pl.Path("./schemas") / (args.input_dataset + "_schema.csv")
    out_schema = pl.Path("./schemas") / (args.output_dataset + "_schema.csv")
    print(f"copying schema file {in_schema} to {out_schema}...", end="")
    shutil.copy(in_schema, out_schema)
    print("done!")
