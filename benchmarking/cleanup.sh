#!/bin/bash

# Files and directories to be removed
FILES=("noir_output.txt" "quantized.txt")
DIRECTORIES=("noir_project/target" "datasets")

# Remove specified files
echo "Removing files..."
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        rm "$file"
        echo "Removed $file"
    else
        echo "File $file does not exist."
    fi
done

# Remove specified directories
echo "Removing directories..."
for dir in "${DIRECTORIES[@]}"; do
    if [ -d "$dir" ]; then
        rm -rf "$dir"
        echo "Removed $dir"
    else
        echo "Directory $dir does not exist."
    fi
done

echo "Cleanup completed."
