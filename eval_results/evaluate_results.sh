#!/bin/bash

FOLDER_PATH="/Users/mariadancianu/Desktop/Git Projects/SQuAD_RAG_experiments/eval_results/test_new_class"

for file in "$FOLDER_PATH"/pred_*.json; do
    if [[ -f "$file" ]]; then
        out_file="${FOLDER_PATH}/eval_$(basename "$file")"
        echo "Processing: $file -> Output: $out_file"
        python evaluation.py data_updated_500.json "$file" --out-file "$out_file"

    fi
done
