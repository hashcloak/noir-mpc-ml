from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd
import numpy as np

if __name__ == "__main__":
    dataset = pd.read_csv("./datasets/train_data_original.csv", index_col=0)
    print("Dataset")
    print(dataset.head())

    model = OneVsRestClassifier(LogisticRegression(penalty=None))
    model.fit(
        dataset[
            [
                "sepal length (cm)",
                "sepal width (cm)",
                "petal length (cm)",
                "petal width (cm)",
            ]
        ],
        dataset["species"],
    )

    test_dataset = pd.read_csv("./datasets/test_data.csv", index_col=0)

    accuracy = model.score(
        test_dataset[
            [
                "sepal length (cm)",
                "sepal width (cm)",
                "petal length (cm)",
                "petal width (cm)",
            ]
        ],
        test_dataset["species"],
    )

    model_idx = 0
    for estimator in model.estimators_:
        print("Model", model_idx)
        print("Weights:", estimator.coef_)
        print("Bias", estimator.intercept_)
        model_idx += 1

    print("accuracy =", accuracy)
