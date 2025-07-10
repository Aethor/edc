import argparse, shutil
import pathlib as pl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dataset", type=str)
    parser.add_argument("-o", "--output-dataset", type=str)
    parser.add_argument("-oy", "--old-year", type=str)
    parser.add_argument("-ny", "--new-year", type=str)
    args = parser.parse_args()

    with open(pl.Path("./dsets") / (args.input_dataset + ".txt")) as f:
        fact_descs = f.readlines()
    fact_descs = [desc.replace(args.old_year, args.new_year) for desc in fact_descs]

    with open(pl.Path("./evaluate/references") / (args.input_dataset + ".txt")) as f:
        refs = f.readlines()
    assert len(fact_descs) == len(refs)
    refs = [r.replace(args.old_year, args.new_year) for r in refs]

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
