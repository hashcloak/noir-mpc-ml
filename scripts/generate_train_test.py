"""
This script loads the Iris dataset and takes the samples labeled as 0 and 1.
Then it splits the dataset according to the SAMPLES_TRAIN and SAMPLES_TEST and
stored the splitted dataset into CSV files in the datasets/ folder.
"""

from sklearn import model_selection
from sklearn import datasets
from sklearn import utils
from sklearn import preprocessing
import pandas as pd

SAMPLES_TRAIN = 30
SAMPLES_TEST = 20

if __name__ == "__main__":
    # Loads the dataset.
    iris = datasets.load_iris()
    iris_dataset = pd.DataFrame(data=iris.data, columns=iris.feature_names)

    # Add the target column as species
    iris_dataset["species"] = iris.target

    # Select SAMPLES_TRAIN + SAMPLES_TEST random elements from the dataset.
    dataset = utils.resample(iris_dataset, n_samples=SAMPLES_TRAIN + SAMPLES_TEST)
    print("DATASET")
    print(dataset)

    # Divide the dataset into train and test without binarizing.
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        dataset.iloc[:, :4],
        dataset.iloc[:, -1],
        train_size=SAMPLES_TRAIN,
        random_state=1,
    )

    # Assemble test dataset and save it into a file without binarizing the
    # response variable.
    train_df = pd.DataFrame(X_train, columns=iris.feature_names)
    train_df["species"] = y_train
    test_df = pd.DataFrame(X_test, columns=iris.feature_names)
    test_df["species"] = y_test
    test_df.to_csv("datasets/test_data.csv")

    print("TRAIN DATASET")
    print(train_df)
    print("TEST DATASET")
    print(test_df)

    # Save the train dataset before binarizing the response variable
    train_df.to_csv("./datasets/train_data_original.csv")

    # Binarizing the target.
    label_binarizer = preprocessing.LabelBinarizer()
    label_binarizer_output = label_binarizer.fit_transform(train_df["species"])
    label_df = pd.DataFrame(
        label_binarizer_output, columns=label_binarizer.classes_, index=train_df.index
    )

    print("LABELS")
    print(label_df)

    train_df_binarized = pd.concat([train_df, label_df], axis=1)
    train_df_binarized.drop(columns=["species"], inplace=True)
    print("BINARIZED DATASET")
    print(train_df_binarized)

    train_df_binarized.to_csv("./datasets/train_data.csv", index=False)
