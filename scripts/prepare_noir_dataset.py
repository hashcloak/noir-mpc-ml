"""
This script reads the train dataset from the datasets/ folder and convert it
into Noir format. The output is provided through the stdout.
"""

import pandas as pd

SCALE = 2**16

if __name__ == "__main__":
    dataset = pd.read_csv("./datasets/train_data.csv")

    print("[")
    for i in range(dataset.shape[0]):
        print("\t[")
        for j in range(4):
            print("\t\tQuantized { x:", round(dataset.iloc[i, j] * SCALE), "}, ")
        print("\t],")
    print("]")

    print("================== LABEL 0 ===========================")

    print("[")
    for i in range(dataset.shape[0]):
        print("\tQuantized { x:", int(dataset.iloc[i, -3] * SCALE), "}, ")
    print("]")

    print("================== LABEL 1 ===========================")

    print("[")
    for i in range(dataset.shape[0]):
        print("\tQuantized { x:", int(dataset.iloc[i, -2] * SCALE), "}, ")
    print("]")

    print("================== LABEL 2 ===========================")

    print("[")
    for i in range(dataset.shape[0]):
        print("\tQuantized { x:", int(dataset.iloc[i, -1] * SCALE), "}, ")
    print("]")
