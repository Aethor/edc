#!/bin/bash

for dataset in 'yago2022:balanced-yago2026' 'yago2026:balanced-yago2022' 'yago2022_multi:balanced-yago2026_multi' 'yago2026_multi:balanced-yago2022_multi' 'yago2022:balanced-yago2026:retimestamped-2026' 'yago2026:balanced-yago2022:retimestamped-2022' 'yago2022_multi:balanced-yago2026_multi:retimestamped-2026' 'yago2026_multi:balanced-yago2022_multi:retimestamped-2022'; do

    python run_baseline.py \
           --input_text_file_path "./dsets/${dataset}.txt" \
           --llm 'mistralai/Mistral-7B-Instruct-v0.2' \
           --cie_prompt_template_file_path './prompt_templates/cie_template.txt' \
           --cie_few_shot_examples_file_path "./few_shot_examples/${dataset}/oie_few_shot_examples.txt" \
           --target_schema_path "./schemas/${dataset}_schema.csv" \
           --output_dir "./output/${dataset}_baseline_target_alignment"

done
