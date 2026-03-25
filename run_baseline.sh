#!/bin/bash


models=('hf:mistralai/Mistral-7B-Instruct-v0.2' 'hf:meta-llama/Llama-3.1-8B-Instruct')

datasets=('yago2019:balanced-yago2026' 'yago2026:balanced-yago2019' 'yago2019_multi:balanced-yago2026_multi' 'yago2026_multi:balanced-yago2019_multi' 'yago2019:balanced-yago2026:retimestamped-2026' 'yago2026:balanced-yago2019:retimestamped-2019' 'yago2019_multi:balanced-yago2026_multi:retimestamped-2026' 'yago2026_multi:balanced-yago2019_multi:retimestamped-2019')

for model in "${models[@]}"; do

    for dataset in "${datasets[@]}"; do

        safe_model_name=$(echo "${model}" | tr '/' ':')

        python run_baseline.py \
            --input_text_file_path "./dsets/${dataset}.txt" \
            --llm "${model}" \
            --cie_prompt_template_file_path './prompt_templates/cie_template.txt' \
            --cie_few_shot_examples_file_path "./few_shot_examples/${dataset}/oie_few_shot_examples.txt" \
            --target_schema_path "./schemas/${dataset}_schema.csv" \
            --output_dir "./output/baseline/${safe_model_name}/${dataset}_target_alignment"

    done

done
