#!/bin/bash

# this creates:
# - yago2022:balanced-yago2026
# - yago2026:balanced-yago2022
python balanced_datasets.py\
       --first-dataset "yago2022"\
       --second-dataset "yago2026"

# this creates:
# - yago2022_multi:balanced-yago2026_multi
# - yago2026_multi:balanced-yago2022_multi
python balanced_datasets.py\
       --first-dataset "yago2022_multi"\
       --second-dataset "yago2026_multi"

for dataset in 'yago2022:balanced-yago2026' 'yago2026:balanced-yago2022' 'yago2022_multi:balanced-yago2026_multi' 'yago2026_multi:balanced-yago2022_multi'; do

    python run.py \
        --oie_llm mistralai/Mistral-7B-Instruct-v0.2 \
        --oie_few_shot_example_file_path "./few_shot_examples/${dataset}/oie_few_shot_examples.txt" \
        --sd_llm gpt-3.5-turbo \
        --sd_few_shot_example_file_path "./few_shot_examples/${dataset}/sd_few_shot_examples.txt" \
        --sc_llm gpt-3.5-turbo \
        --sc_embedder intfloat/e5-mistral-7b-instruct \
        --input_text_file_path "./dsets/${dataset}.txt" \
        --target_schema_path "./schemas/${dataset}_schema.csv" \
        --logging_verbose \
        --output_dir "./output/${dataset}_target_alignment"

done


python retimestamped_dataset.py\
       --input-dataset "yago2022:balanced-yago2026"\
       --output-dataset "yago2022:balanced-yago2026:retimestamped-2026"\
       --old-year 2022\
       --new-year 2026

python retimestamped_dataset.py\
       --input-dataset "yago2026:balanced-yago2022"\
       --output-dataset "yago2026:balanced-yago2022:retimestamped-2022"\
       --old-year 2026\
       --new-year 2022

python retimestamped_dataset.py\
       --input-dataset "yago2022_multi:balanced-yago2026_multi"\
       --output-dataset "yago2022_multi:balanced-yago2026_multi:retimestamped-2026"\
       --old-year 2022\
       --new-year 2026

python retimestamped_dataset.py\
       --input-dataset "yago2026_multi:balanced-yago2022_multi"\
       --output-dataset "yago2026_multi:balanced-yago2022_multi:retimestamped-2022"\
       --old-year 2026\
       --new-year 2022

for dataset in 'yago2022:balanced-yago2026:retimestamped-2026' 'yago2026:balanced-yago2022:retimestamped-2022' 'yago2022_multi:balanced-yago2026_multi:retimestamped-2026' 'yago2026_multi:balanced-yago2022_multi:retimestamped-2022'; do

    python run.py \
        --oie_llm mistralai/Mistral-7B-Instruct-v0.2 \
        --oie_few_shot_example_file_path "./few_shot_examples/${dataset}/oie_few_shot_examples.txt" \
        --sd_llm gpt-3.5-turbo \
        --sd_few_shot_example_file_path "./few_shot_examples/${dataset}/sd_few_shot_examples.txt" \
        --sc_llm gpt-3.5-turbo \
        --sc_embedder intfloat/e5-mistral-7b-instruct \
        --input_text_file_path "./dsets/${dataset}.txt" \
        --target_schema_path "./schemas/${dataset}_schema.csv" \
        --logging_verbose \
        --output_dir "./output/${dataset}_target_alignment"

done
