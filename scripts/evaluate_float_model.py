import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from typing import List

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
        for model in self.models:
            print(model.weights)
            print(model.bias)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    file = open("scripts/float.txt")

    lines = file.readlines()
    models = []

    for c in range(CLASSES):
        parameters = []
        for m in range(FEATURES + 1):
            float_val = float(lines[(FEATURES + 1) * c + m])
            print("Value read:", float_val)

            # Check if the values are negative and convert them.
            parameters.append(float_val)

        model = Model(parameters[: len(parameters) - 1], parameters[-1])
        models.append(model)

    multi_model = MultiClassModel(models)
    multi_model.print_params()

    # Load test model.
    dataset = pd.read_csv("./datasets/test_data.csv", index_col=0)
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
