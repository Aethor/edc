#!/bin/bash

models=('hf:mistralai/Mistral-7B-Instruct-v0.2' 'hf:meta-llama/Llama-3.1-8B-Instruct')

datasets=('yago2019:balanced-yago2026' 'yago2026:balanced-yago2019' 'yago2019_multi:balanced-yago2026_multi' 'yago2026_multi:balanced-yago2019_multi' 'yago2019:balanced-yago2026:retimestamped-2026' 'yago2026:balanced-yago2019:retimestamped-2019' 'yago2019_multi:balanced-yago2026_multi:retimestamped-2026' 'yago2026_multi:balanced-yago2019_multi:retimestamped-2019')

for model in "${models[@]}"; do

    for dataset in "${datasets[@]}"; do

        safe_model_name=$(echo "${model}" | tr '/' ':')
        output_dir="./output/edc/${safe_model_name}"

        echo -n "scoring ${dataset}..."
        python -m evaluate.evaluation_script\
            --edc_output "${output_dir}/${dataset}_target_alignment/iter0/canon_kg.txt"\
            --reference "./evaluate/references/${dataset}.txt"\
            > "${output_dir}/${dataset}_score.txt"
        echo "done!"

    done

done
