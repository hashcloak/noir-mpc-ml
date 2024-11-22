"""
This script performs the following steps:
    - Loads the parameters of the trained model from the quantized.txt file.
    - Loads the test dataset from the dataset folder.
    - Computes the accuracy of the trained model with respect to the test
      dataset.

The quantized.txt file should contain the parameters in hexadecimal notation
each of them in a different line. The file will list the weights of the trained
model first and the last parameter will be the bias.
"""

import numpy as np
import pandas as pd

POWER_SCALE = 16
MODULUS = 21888242871839275222246405745257275088548364400416034343698204186575808495617

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if __name__ == "__main__":
    file = open("scripts/quantized.txt")

    parameters = []
    for line in file.readlines():
        int_rep = int(line, base=0)
        print("Field value:", int_rep)

        # Check if the values are negative and convert them.
        most_sig_bytes = int_rep >> 16 * 8
        if most_sig_bytes != 0:
            int_rep = -(MODULUS - int_rep)

        float_value = int_rep * (2 ** -POWER_SCALE)
        print("Float value:", float_value)

        parameters.append(float_value)

    bias = parameters[-1]
    weights = np.array(parameters[:len(parameters)-1])
    weights = weights.reshape((len(weights), 1))

    print("Weights:", weights)
    print("Bias:", bias)

    # Load test model.
    dataset = pd.read_csv("./datasets/test_data.csv")
    test_data = dataset.iloc[:, :4].to_numpy()
    response_var = dataset.iloc[:, -1].to_numpy().reshape((dataset.shape[0], 1))

    # Evaluate the test model.
    evaluations = sigmoid(np.dot(test_data, weights) + bias)
    evaluations = np.where(evaluations < 0.5, 0, 1)

    print("Evaluations:", evaluations)

    # Compute the accuracy
    accuracy = (1 / test_data.shape[0]) * np.sum(evaluations == response_var)
    print("Accuracy:", accuracy.item())
    







