# An Adapted Version of EDC for Temporal Knowledge Graph Extraction 

The original "Extract, Define, Canonicalize" (EDC) framework can only extract semantic triples `(subject, relation, object)`. In this repository, we extend the [original version](https://github.com/clear-nus/edc) to extract quadruples of the form `(subject, relation, object, timestamp)`.



# Installing Dependencies

We use the `uv` package manager to deal with dependencies. You can install them using:

```sh
uv sync
```

Alternatively, you can also use pip directly (you might want to create and activate a virtual environment first):

```sh
pip install -r requirements.txt
```



# Reproducing Experiments

## Running Benchmarks

The `run_all.sh` script will reproduce the experiments from the paper:

```sh
for dataset in 'yago2022:balanced-yago2026' 'yago2026:balanced-yago2022' 'yago2022_multi:balanced-yago2026_multi' 'yago2026_multi:balanced-yago2022_multi' 'yago2022:balanced-yago2026:retimestamped-2026' 'yago2026:balanced-yago2022:retimestamped-2022' 'yago2022_multi:balanced-yago2026_multi:retimestamped-2026' 'yago2026_multi:balanced-yago2022_multi:retimestamped-2022'; do

    python run.py \
           --oie_llm 'mistralai/Mistral-7B-Instruct-v0.2' \
           --oie_few_shot_example_file_path "./few_shot_examples/${dataset}/oie_few_shot_examples.txt" \
           --sd_llm gpt-3.5-turbo \
           --sd_few_shot_example_file_path "./few_shot_examples/${dataset}/sd_few_shot_examples.txt" \
           --sc_llm gpt-3.5-turbo \
           --sc_embedder intfloat/e5-mistral-7b-instruct \
           --input_text_file_path "./dsets/${dataset}.txt" \
           --target_schema_path "./schemas/${dataset}_schema.csv" \
           --logging_verbose \
           --output_dir "./output/${dataset}_target_alignment"

    python run_baseline.py \
           --input_text_file_path "./dsets/${dataset}.txt" \
           --llm 'mistralai/Mistral-7B-Instruct-v0.2' \
           --cie_prompt_template_file_path './prompt_templates/cie_template.txt' \
           --cie_few_shot_examples_file_path "./few_shot_examples/${dataset}/oie_few_shot_examples.txt" \
           --target_schema_path "./schemas/${dataset}_schema.csv" \
           --output_dir "./output/${dataset}_baseline_target_alignment"

done
```

For a dataset `$dataset`, an invocation of the `run.py` script will create two files:

- `./output/$dataset_target_alignment/iter0/canon_kg.txt`
- `./output/$dataset_target_alignment/iter0/result_at_each_stage.json`


## Scoring Benchmarks

The `score_all.sh` script will compute the scores for a previously ran benchmark. For each dataset `$dataset`, the script will create a file `./output/$dataset_score.txt`.

In details, the `score_all.sh` runs the `evaluate/evaluation_script.py` module:

```sh
datasets=('yago2022:balanced-yago2026' 'yago2026:balanced-yago2022' 'yago2022_multi:balanced-yago2026_multi' 'yago2026_multi:balanced-yago2022_multi' 'yago2022:balanced-yago2026:retimestamped-2026' 'yago2026:balanced-yago2022:retimestamped-2022' 'yago2022_multi:balanced-yago2026_multi:retimestamped-2026' 'yago2026_multi:balanced-yago2022_multi:retimestamped-2022')

for dataset in "${datasets[@]}"; do
    echo -n "scoring ${dataset}..."
    python -m evaluate.evaluation_script\
           --edc_output "./output/${dataset}_target_alignment/iter0/canon_kg.txt"\
           --reference "./evaluate/references/${dataset}.txt"\
           > "./output/${dataset}_score.txt"
    echo "done!"
done
```

This module originally comes from the [WebNLG 2020 text-to-triples evaluation script](https://github.com/WebNLG/WebNLG-Text-to-triples/tree/ea436d431752e7a033741bbf7b0120930847dd77). However, we found some bugs in the original implementation, as highlighted in the *Technical Appendix* of our article. We corrected these bugs, and performed property-based testing to ensure correctness (see `./tests/test_evaluaterefcand.py`). The original implementation it available at `./evaluate/archive.py`.


# Details

## Dataset Naming

Each dataset has a name composed by a base name and a series of additions that reflects the operation it went through, separated by colons `:`. For example, the name `yago2026:balanced-yago2022:retimestamped-2022` indicates that the initial `yago2026` dataset was first balanced to have the same number of relationships as `yago2022`, and that its timestamps where changed to 2022. The scripts responsible for these transformations are `balanced_datasets.py` and `retimestamped_dataset.py`.

## Dataset Structure

Each dataset `$dataset` is composed of the following files:

- `./dsets/$dataset.txt`: The text from which to extract quadruples.
- `./few_shot_examples/$dataset/oie_few_shot_examples.txt`: Few-shots examples for the "Extract" phase.
- `./few_shot_examples/$dataset/sd_few_shot_examples.txt`: Few-shots examples for the "Define" phase.
- `./schemas/$dataset_schema.csv`: Relation descriptions.
- `./evaluate/references/$dataset.txt`: Reference quadruples for each input text.

## Utilities

There are some utility scripts available:

- `dataset2edc.py`: convert a dataset of generated facts and description into the edc format.
- `balanced_datasets.py`: create two new datasets by balancing the relation distribution between two datasets.
- `retimestamped_dataset.py`: create a new dataset by modifying the timestamps in a dataset.
