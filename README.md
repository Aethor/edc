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

The `run_edc.sh` and `run_baseline.sh` scripts will reproduce the experiments from the paper. Outputs will be saved under `./output/{edc,baseline}/${model}/${dataset}_target_alignment`.


## Scoring Benchmarks

The `score_edc.sh` and `score_baseline.sh` scripts will compute the scores for previously ran experiments. For each dataset `$dataset`, scripts will create a file `./output/{edc,baseline}/${model}/$dataset_score.txt`.


### Note on scoring

The scoring script run the `evaluate/evaluation_script.py` module. This module originally comes from the [WebNLG 2020 text-to-triples evaluation script](https://github.com/WebNLG/WebNLG-Text-to-triples/tree/ea436d431752e7a033741bbf7b0120930847dd77). However, we found some bugs in the original implementation, as highlighted in the appendix of our article. We corrected these bugs, and performed property-based testing to ensure correctness (see `./tests/test_evaluaterefcand.py`). The original implementation it available at `./evaluate/archive.py`.


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
