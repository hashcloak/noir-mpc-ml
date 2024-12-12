#!/bin/bash

# Default values
DATASET_NAME="iris"
PROJECT_DIR="./noir_project"
TARGET_DIR="$PROJECT_DIR/target"
OUT_DIR="output"
OUTPUT_BENCH="$OUT_DIR/benchmarks.txt"

# Ranges for parameters
EPOCHS_LIST=(5 10 15)
SAMPLES_TRAIN_LIST=(10 20 30)

# Ensure output directory exists
if [ ! -d "$OUT_DIR" ]; then
    mkdir -p "$OUT_DIR"
fi

# Clear the benchmark file
> "$OUTPUT_BENCH"

# Loop through parameter combinations
for EPOCHS in "${EPOCHS_LIST[@]}"; do
    for SAMPLES_TRAIN in "${SAMPLES_TRAIN_LIST[@]}"; do
        echo "Running benchmark with epochs=$EPOCHS, samples_train=$SAMPLES_TRAIN"

        # Step 1: Generate dataset
        echo "Generating dataset..."
        python3 generate_dataset.py --dataset "$DATASET_NAME" --samples-train "$SAMPLES_TRAIN"
        if [ $? -ne 0 ]; then
            echo "Error: Dataset generation failed for epochs=$EPOCHS, samples_train=$SAMPLES_TRAIN."
            continue
        fi

        # Step 2: Load metadata
        METADATA_FILE="./datasets/metadata.json"
        TRAIN_DATA_FILE="./datasets/train_data.json"
        if [ ! -f "$METADATA_FILE" ] || [ ! -f "$TRAIN_DATA_FILE" ]; then
            echo "Error: Metadata or training data file not found for epochs=$EPOCHS, samples_train=$SAMPLES_TRAIN."
            continue
        fi

        # Step 3: Generate Noir files
        echo "Generating Noir main.nr and Prover.toml..."
        python3 write_noir_main.py --metadata "$METADATA_FILE" --data "$TRAIN_DATA_FILE" --epochs "$EPOCHS" --samples-train "$SAMPLES_TRAIN" --output-dir "$PROJECT_DIR"
        if [ $? -ne 0 ]; then
            echo "Error: Noir file generation failed for epochs=$EPOCHS, samples_train=$SAMPLES_TRAIN."
            continue
        fi

        # Step 4: Compile the Noir project
        echo "Compiling Noir project..."
        if [ ! -d "$PROJECT_DIR" ]; then
            echo "Error: Project directory $PROJECT_DIR does not exist!"
            continue
        fi

        pushd "$PROJECT_DIR" > /dev/null
        nargo execute
        if [ $? -ne 0 ]; then
            echo "Error: Failed to compile the Noir project for epochs=$EPOCHS, samples_train=$SAMPLES_TRAIN."
            popd > /dev/null
            continue
        fi
        popd > /dev/null

        # Step 5: Check for compiled output
        echo "Checking for compiled output..."
        if [ ! -f "$TARGET_DIR/noir_project.json" ]; then
            echo "Error: Compiled file noir_project.json not found in $TARGET_DIR for epochs=$EPOCHS, samples_train=$SAMPLES_TRAIN."
            continue
        fi

        # Step 6: Get gatecount for current params
        echo "Getting gatecount..."
        echo "epochs=$EPOCHS, samples_train=$SAMPLES_TRAIN" >> "$OUTPUT_BENCH"
        bb gates -b "$TARGET_DIR/noir_project.json" >> "$OUTPUT_BENCH"
        echo "===" >> "$OUTPUT_BENCH"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to get gatecount for epochs=$EPOCHS, samples_train=$SAMPLES_TRAIN."
            continue
        fi

        echo "Gatecount completed for epochs=$EPOCHS, samples_train=$SAMPLES_TRAIN."
    done
done

# Step 7: Generate CSV from benchmarks
python3 generate_gatecounts_csv.py

echo "Benchmarking completed. CSV file created at benches/benchmarks.csv."
