#!/bin/bash

datasets=('yago2022:balanced-yago2026' 'yago2026:balanced-yago2022' 'yago2022_multi:balanced-yago2026_multi' 'yago2026_multi:balanced-yago2022_multi' 'yago2022:balanced-yago2026:retimestamped-2026' 'yago2026:balanced-yago2022:retimestamped-2022' 'yago2022_multi:balanced-yago2026_multi:retimestamped-2026' 'yago2026_multi:balanced-yago2022_multi:retimestamped-2022')

model='mistralai/Mistral-7B-Instruct-v0.2'

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
