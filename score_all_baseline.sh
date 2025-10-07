#!/bin/bash

datasets=('yago2022:balanced-yago2026' 'yago2026:balanced-yago2022' 'yago2022_multi:balanced-yago2026_multi' 'yago2026_multi:balanced-yago2022_multi')

for dataset in "${datasets[@]}"; do
    echo -n "scoring ${dataset}..."
    python -m evaluate.evaluation_script\
           --edc_output "./output/${dataset}_baseline_target_alignment/iter0/canon_kg.txt"\
           --reference "./evaluate/references/${dataset}.txt"\
           > "./output/${dataset}_score.txt"
    echo "done!"
done
