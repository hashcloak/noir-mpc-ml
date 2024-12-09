from typing import List
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import argparse

POWER_SCALE = 16
MODULUS = 21888242871839275222246405745257275088548364400416034343698204186575808495617


class Model:
    def __init__(self, weights: List[float], bias: float):
        self.weights = np.array(weights).reshape((len(weights), 1))
        self.bias = bias

    def predict_probability(self, X):
        return sigmoid(np.dot(X, self.weights) + self.bias)

    def predict(self, X):
        evaluations = self.predict_probability(X)
        return np.where(evaluations < 0.5, 0, 1)


class MultiClassModel:
    def __init__(self, models: List[Model]):
        self.models = models

    def predict(self, X):
        predictions = []
        for model in self.models:
            predictions.append(model.predict_probability(X))
        return np.argmax(np.column_stack(predictions), axis=1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def load_quantized_params(file, features, classes):
    models = []
    with open(file, "r") as f:
        lines = f.readlines()

    for c in range(classes):
        start = c * (features + 1)
        weights = []
        for i in range(features):
            weights.append(decode_quantized(lines[start + i].strip()))
        bias = decode_quantized(lines[start + features].strip())
        models.append(Model(weights, bias))

    return MultiClassModel(models)


def decode_quantized(hex_str):
    int_rep = int(hex_str, 16)
    most_sig_bytes = int_rep >> (16 * 8)
    if most_sig_bytes != 0:
        int_rep = -(MODULUS - int_rep)
    return int_rep * (2 ** -POWER_SCALE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute model accuracy based on Noir outputs.")
    parser.add_argument("--quantized-file", required=True, help="Quantized parameters file")
    parser.add_argument("--test-data", required=True, help="Test dataset CSV file")
    parser.add_argument("--classes", type=int, required=True, help="Number of classes")
    parser.add_argument("--features", type=int, required=True, help="Number of features")
    parser.add_argument("--samples", type=int, required=True, help="Number of test samples")
    args = parser.parse_args()

    # Load quantized parameters
    multi_model = load_quantized_params(args.quantized_file, args.features, args.classes)

    # Load test dataset
    dataset = pd.read_csv(args.test_data)
    test_data = dataset.iloc[:, :-1].to_numpy()
    response_var = dataset.iloc[:, -1].to_numpy()

    if test_data.shape[1] != args.features:
        raise ValueError(f"Expected {args.features} features, but got {test_data.shape[1]}")
    if len(response_var) != args.samples:
        raise ValueError(f"Expected {args.samples} samples, but got {len(response_var)}")

    # Compute predictions and accuracy
    predictions = multi_model.predict(test_data)
    accuracy = accuracy_score(response_var, predictions)

    print(f"Accuracy: {accuracy:.4f}")
