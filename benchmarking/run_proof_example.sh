#!/bin/bash

# Default values
EPOCHS=10
SAMPLES_TRAIN=30
SAMPLES_TEST=20
DATASET_NAME="iris"
PROJECT_DIR="./noir_project"
TARGET_DIR="$PROJECT_DIR/target"

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

# Step 6: Generate the proof
echo "Generating proof..."
bb prove -b "$TARGET_DIR/noir_project.json" -w "$TARGET_DIR/noir_project.gz" -o "$TARGET_DIR/proof"
if [ $? -ne 0 ]; then
    echo "Error: Failed to generate proof."
    exit 1
fi

# Step 7: Write verification key
echo "Writing verification key..."
bb write_vk -b "$TARGET_DIR/noir_project.json" -o "$TARGET_DIR/vk"
if [ $? -ne 0 ]; then
    echo "Error: Failed to write verification key."
    exit 1
fi

# Step 8: Verify the proof
echo "Verifying proof..."
bb verify -k "$TARGET_DIR/vk" -p "$TARGET_DIR/proof"
if [ $? -ne 0 ]; then
    echo "Error: Proof verification failed."
    exit 1
fi

echo "Proof successfully verified."
