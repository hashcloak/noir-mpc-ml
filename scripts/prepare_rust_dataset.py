"""
This script reads the train dataset from the datasets/ folder and converts it
into Rust format. The output is provided through the stdout.
"""

import pandas as pd

if __name__ == "__main__":
    dataset = pd.read_csv("./datasets/train_data.csv")

    # Print data for Rust
    print("let inputs: [[f64; M]; N] = [")
    for i in range(dataset.shape[0]):
        print("\t[")
        for j in range(4):
            print(f"\t\t{dataset.iloc[i, j]:.6},")  # f64 format
        print("\t],")
    print("];\n")

    print("//================== LABELS ===========================")

    print("let labels_class0 = [")
    for i in range(dataset.shape[0]):
        print(f"\t{dataset.iloc[i, -3]:.1f},")  # Force 0.0 or 1.0 format
    print("];")

    print("//================== LABELS ===========================")

    print("let labels_class1 = [")
    for i in range(dataset.shape[0]):
        print(f"\t{dataset.iloc[i, -2]:.1f},")  # Force 0.0 or 1.0 format
    print("];")

    print("//================== LABELS ===========================")

    print("let labels_class2 = [")
    for i in range(dataset.shape[0]):
        print(f"\t{dataset.iloc[i, -1]:.1f},")  # Force 0.0 or 1.0 format
    print("];")
