#!/bin/bash

# Set FOLDER_PATH to the current directory + /eval_res
FOLDER_PATH=$(pwd)/test_new_class

echo "FOLDER_PATH: $FOLDER_PATH"

for file in "$FOLDER_PATH"/pred_*.json; do
    if [[ -f "$file" ]]; then
        out_file="${FOLDER_PATH}/eval_$(basename "$file")"
        echo "Processing: $file -> Output: $out_file"
        python evaluation.py data_updated_500.json "$file" --out-file "$out_file"

    fi
done
