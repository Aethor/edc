#!/bin/bash

models=('mistralai/Mistral-7B-Instruct-v0.2' 'meta-llama/Llama-3.1-8B')

datasets=('yago2022:balanced-yago2026' 'yago2026:balanced-yago2022' 'yago2022_multi:balanced-yago2026_multi' 'yago2026_multi:balanced-yago2022_multi' 'yago2022:balanced-yago2026:retimestamped-2026' 'yago2026:balanced-yago2022:retimestamped-2022' 'yago2022_multi:balanced-yago2026_multi:retimestamped-2026' 'yago2026_multi:balanced-yago2022_multi:retimestamped-2022')

for model in "${models[@]}"; do

    for dataset in "${datasets[@]}"; do

        safe_model_name=$(echo "${model}" | tr '/' ':')

        echo -n "scoring ${dataset}..."
        python -m evaluate.evaluation_script\
            --edc_output "./output/baselines/${safe_model_name}/${dataset}_target_alignment/iter0/canon_kg.txt"\
            --reference "./evaluate/references/${dataset}.txt"\
            > "./output/baselines/${safe_model_name}/${dataset}_score.txt"
        echo "done!"
    done

done
