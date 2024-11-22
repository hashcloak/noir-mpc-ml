
import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if __name__ == "__main__":
    file = open("scripts/float.txt")

    parameters = []
    for line in file.readlines():
        float_value = float(line)

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
    print("Response variable test dataset:", response_var)

    # Evaluate the test model.
    evaluations = sigmoid(np.dot(test_data, weights) + bias)
    evaluations = np.where(evaluations < 0.5, 0, 1)

    print("Evaluations:", evaluations)

    # Compute the accuracy
    accuracy = (1 / test_data.shape[0]) * np.sum(evaluations == response_var)
    print("Accuracy:", accuracy.item())