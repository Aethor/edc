import random, argparse, shutil
import pathlib as pl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dataset", type=str)
    parser.add_argument("-o", "--output-dataset", type=str)
    parser.add_argument("-p", "--proportion", type=float, help="between 0.0 and 1.0")
    args = parser.parse_args()
    assert args.proportion > 0.0 and args.proportion <= 1.0

    random.seed(0)

    with open(pl.Path("./dsets") / (args.input_dataset + ".txt")) as f:
        fact_descs = f.readlines()

    with open(pl.Path("./evaluate/references") / (args.input_dataset + ".txt")) as f:
        refs = f.readlines()
    assert len(fact_descs) == len(refs)

    examples_nb = int(args.proportion * len(refs))
    print(f"extracting a subdataset of {examples_nb} examples.")
    example_indices = random.sample(
        list(range(len(fact_descs))), k=int(args.proportion * len(refs))
    )

    out_fact_descs_path = pl.Path("./dsets") / (args.output_dataset + ".txt")
    print(f"writing {out_fact_descs_path}...", end="")
    with open(out_fact_descs_path, "w") as f:
        out_facts = [fact_descs[i] for i in example_indices]
        out_facts[-1] = out_facts[-1].rstrip("\n")
        f.writelines(out_facts)
    print("done!")

    refs_path = pl.Path("./evaluate/references") / (args.output_dataset + ".txt")
    print(f"writing {refs_path}...", end="")
    with open(refs_path, "w") as f:
        out_refs = [refs[i] for i in example_indices]
        out_refs[-1] = out_refs[-1].rstrip("\n")
        f.writelines(out_refs)
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
