import csv, os
from argparse import ArgumentParser
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from edc.extract import Extractor

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input_text_file_path",
        default="./datasets/example.txt",
        help="File containing input texts to extract KG from, each line contains one piece of text.",
    )
    parser.add_argument(
        "--llm",
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="LLM used for information extraction",
    )
    parser.add_argument(
        "--cie_prompt_template_file_path",
        type=str,
        default="./prompt_templates/cie_template.txt",
    )
    parser.add_argument(
        "--cie_few_shot_examples_file_path",
        type=str,
        default="./few_shot_examples/example/oie_few_shot_examples.txt",
    )
    parser.add_argument(
        "--target_schema_path",
        default="./schemas/example_schema.csv",
        help="File containing the schema relations.",
    )
    parser.add_argument(
        "--output_dir", default="./output/tmp", help="Directory to output to."
    )
    args = parser.parse_args()

    with open(args.input_text_file_path) as f:
        input_text_list = f.readlines()

    relations = set()
    with open(args.target_schema_path) as f:
        reader = csv.reader(f)
        for row in reader:
            # we explicitly ignore the relation description because of
            # context length
            relation, _ = row
            relations.add(relation)

    with open(args.cie_prompt_template_file_path) as f:
        prompt_template = f.read()

    with open(args.cie_few_shot_examples_file_path) as f:
        examples = f.read()

    model = AutoModelForCausalLM.from_pretrained(
        args.llm, dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    extractor = Extractor(model, tokenizer)

    os.makedirs(f"{args.output_dir}/iter0", exist_ok=True)
    with open(f"{args.output_dir}/iter0/canon_kg.txt", "w") as f:
        for i, text in enumerate(tqdm(input_text_list)):
            quads = extractor.extract(
                text, examples, prompt_template, relations_hint=", ".join(relations)
            )
            f.write(str(quads))
            if i != len(quads) - 1:
                f.write("\n")
            f.flush()
