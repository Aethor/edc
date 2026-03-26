#!/bin/bash


models=('hf:mistralai/Mistral-7B-Instruct-v0.2' 'hf:meta-llama/Llama-3.1-8B-Instruct')

datasets=('yago_past:balanced-yago2026' 'yago2026:balanced-yago_past' 'yago_past_multi:balanced-yago2026_multi' 'yago2026_multi:balanced-yago_past_multi' 'yago2026:balanced-yago_past:retimestamped-2022' 'yago2026_multi:balanced-yago_past_multi:retimestamped-2022')

for model in "${models[@]}"; do

    for dataset in "${datasets[@]}"; do

        safe_model_name=$(echo "${model}" | tr '/' ':')
        output_dir="./output/baseline/${safe_model_name}"

        python run_baseline.py \
            --input_text_file_path "./dsets/${dataset}.txt" \
            --llm "${model}" \
            --cie_prompt_template_file_path './prompt_templates/cie_template.txt' \
            --cie_few_shot_examples_file_path "./few_shot_examples/${dataset}/oie_few_shot_examples.txt" \
            --target_schema_path "./schemas/${dataset}_schema.csv" \
            --output_dir "${output_dir}/${dataset}_target_alignment"

        echo -n "scoring ${dataset}..."
        python -m evaluate.evaluation_script\
            --edc_output "./output/baseline/${safe_model_name}/${dataset}_target_alignment/iter0/canon_kg.txt"\
            --reference "./evaluate/references/${dataset}.txt"\
            > "${output_dir}/${dataset}_score.txt"
        echo "done!"

    done

done
