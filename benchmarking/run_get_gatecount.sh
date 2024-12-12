#!/bin/bash

# Default values
EPOCHS=10
SAMPLES_TRAIN=30
SAMPLES_TEST=20
DATASET_NAME="iris"
PROJECT_DIR="./noir_project"
TARGET_DIR="$PROJECT_DIR/target"
OUTPUT_DIR="output"
OUTPUT_BENCH="$OUTPUT_DIR/benchmarks.txt"

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
            DATASET_NAME="${arg#*=}" # Extract value for dataset
            ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

# Ensure output directory exists
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

# Step 1: Generate dataset with dynamic samples
echo "Generating dataset..."
python3 generate_dataset.py --dataset "$DATASET_NAME" --samples-train "$SAMPLES_TRAIN" --samples-test "$SAMPLES_TEST"
if [ $? -ne 0 ]; then
    echo "Error: Dataset generation failed."
    exit 1
fi

# Step 2: Load metadata
METADATA_FILE="./datasets/metadata.json"
TRAIN_DATA_FILE="./datasets/train_data.json"
if [ ! -f "$METADATA_FILE" ] || [ ! -f "$TRAIN_DATA_FILE" ]; then
    echo "Error: Metadata or training data file not found!"
    exit 1
fi

# Step 3: Generate Noir files
echo "Generating Noir main.nr and Prover.toml..."
python3 write_noir_main.py --metadata "$METADATA_FILE" --data "$TRAIN_DATA_FILE" --epochs "$EPOCHS" --samples-train "$SAMPLES_TRAIN" --output-dir "$PROJECT_DIR"
if [ $? -ne 0 ]; then
    echo "Error: Noir file generation failed."
    exit 1
fi

# Step 4: Compile the Noir project
echo "Compiling Noir project..."
if [ ! -d "$PROJECT_DIR" ]; then
    echo "Error: Project directory $PROJECT_DIR does not exist!"
    exit 1
fi

pushd "$PROJECT_DIR" > /dev/null
nargo execute
if [ $? -ne 0 ]; then
    echo "Error: Failed to compile the Noir project."
    popd > /dev/null
    exit 1
fi
popd > /dev/null

# Step 5: Check for compiled output
echo "Checking for compiled output..."
if [ ! -f "$TARGET_DIR/noir_project.json" ]; then
    echo "Error: Compiled file noir_project.json not found in $TARGET_DIR."
    exit 1
fi

# Step 6: Get gatecount
echo "Get gatecount..."
bb gates -b "$TARGET_DIR/noir_project.json" >> "$OUTPUT_BENCH"
if [ $? -ne 0 ]; then
    echo "Error: Failed to get gatecount."
    exit 1
fi

echo "Gatecount finished."
