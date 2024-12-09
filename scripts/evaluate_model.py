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

from typing import List
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


POWER_SCALE = 16
MODULUS = 21888242871839275222246405745257275088548364400416034343698204186575808495617
CLASSES = 3
FEATURES = 4
SAMPLES = 30


class Model:
    def __init__(self, weights: List[float], bias: float):
        self.weights = np.array(weights).reshape((len(weights), 1))
        self.bias = bias

    def predict_probability(self, X):
        """
        Predicts the probability that an element is of class 1.
        """
        return sigmoid(np.dot(X, self.weights) + self.bias)

    def predict(self, X):
        """
        Predicts the class to which the samples belong.
        """
        evaluations = self.predict_probability(X)
        return np.where(evaluations < 0.5, 0, 1)


class MultiClassModel:
    def __init__(self, models: List[Model]):
        self.models = models

    def predict(self, X):
        """
        Returns the class predicted by the model assuming an indexation starting
        with zero. This uses the one-vs-all approach to multiclass prediction.
        """
        predictions = []
        col_num = 0
        for model in self.models:
            pred_model = model.predict_probability(X)
            predictions.append(pd.DataFrame(pred_model, columns=[col_num]))
            col_num += 1
        individual_predictions = pd.concat(predictions, axis=1)
        result = individual_predictions.idxmax(axis=1)
        return result.to_numpy()

    def print_params(self):
        model_idx = 0
        for model in self.models:
            print("Params for model", model_idx)
            print(model.weights)
            print(model.bias)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    file = open("scripts/quantized.txt")

    lines = file.readlines()
    models = []

    for c in range(CLASSES):
        parameters = []
        for m in range(FEATURES + 1):
            int_rep = int(lines[(FEATURES + 1) * c + m], 16)
            print("Field value:", int_rep)

            # Check if the values are negative and convert them.
            most_sig_bytes = int_rep >> (16 * 8)
            if most_sig_bytes != 0:
                int_rep = -(MODULUS - int_rep)

            float_value = int_rep * (2**-POWER_SCALE)
            print("Float value:", float_value)

            parameters.append(float_value)

        model = Model(parameters[: len(parameters) - 1], parameters[-1])
        models.append(model)

    multi_model = MultiClassModel(models)
    print("Model params:")
    multi_model.print_params()

    # Load test model.
    dataset = pd.read_csv("./datasets/test_data.csv", index_col=0)
    print("Test dataset:")
    print(dataset)

    test_data = dataset.iloc[:, :4].to_numpy()
    response_var = dataset.iloc[:, -1].to_numpy()
    print("Response var:", response_var)

    evaluations = multi_model.predict(test_data)
    print("Evaluations:", evaluations)

    # Compute the accuracy
    assert evaluations.shape == response_var.shape
    accuracy = accuracy_score(response_var, evaluations)
    print("Accuracy:", accuracy)
