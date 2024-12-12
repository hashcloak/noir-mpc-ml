#!/bin/bash

# Default values
EPOCHS=10
SAMPLES_TRAIN=30
SAMPLES_TEST=20
DATASET_NAME="iris"

# Parse arguments
for arg in "$@"; do
    case $arg in
        --epochs=*)
            EPOCHS="${arg#*=}" # Extract value for epochs
            ;;
        --samples-train=*)
            SAMPLES_TRAIN="${arg#*=}" # Extract value for training samples
            ;;
        --samples-test=*)
            SAMPLES_TEST="${arg#*=}" # Extract value for test samples
            ;;
        --dataset=*)
            DATASET_NAME="${arg#*=}" # Extract value for test samples
            ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

# Step 1: Generate dataset with dynamic samples
python3 generate_dataset.py --dataset $DATASET_NAME --samples-train $SAMPLES_TRAIN --samples-test $SAMPLES_TEST

# Step 2: Load metadata
METADATA_FILE="./datasets/metadata.json"
if [ ! -f "$METADATA_FILE" ]; then
    echo "Metadata file not found!"
    exit 1
fi

FEATURES=$(jq .features $METADATA_FILE)
CLASSES=$(jq .classes $METADATA_FILE)

# Step 3: Adjust Noir code with dynamic epochs and training samples
python3 write_noir_test.py --epochs $EPOCHS --samples-train $SAMPLES_TRAIN

# Step 4: Run Noir tests
cd noir_project
nargo test --show-output > ../noir_output.txt
cd ..

# Step 5: Parse Noir output
python3 parse_noir_output.py --output-file noir_output.txt --quantized-file quantized.txt --features $FEATURES --classes $CLASSES

# Step 6: Compute accuracy
python3 compute_accuracy.py --quantized-file quantized.txt --test-data ./datasets/test_data.csv --classes $CLASSES --features $FEATURES --samples $SAMPLES_TEST
