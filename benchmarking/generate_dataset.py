import pandas as pd
from sklearn import datasets, model_selection, preprocessing, utils
import argparse
import json
import os

SCALE = 2**16

def quantize_row(features):
    """Quantize the feature row using SCALE."""
    return [round(feature * SCALE) for feature in features]

def load_dataset(dataset_name):
    """Load the specified dataset."""
    if dataset_name == "iris":
        return datasets.load_iris()
    elif dataset_name == "wine":
        return datasets.load_wine()
    elif dataset_name == "digits":
        return datasets.load_digits()
    elif dataset_name == "diabetes":
        return datasets.load_diabetes()
    elif dataset_name == "linnerud":
        return datasets.load_linnerud()
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported. Available options: "
                         f"iris, wine, digits, diabetes, linnerud.")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate dataset with dynamic sample size.")
    parser.add_argument(
        "--dataset", type=str, default="iris",
        help="Dataset to load (default: 'iris')"
    )
    parser.add_argument(
        "--samples-train", type=int, default=30,
        help="Number of training samples (default: 30)"
    )
    parser.add_argument(
        "--samples-test", type=int, default=20,
        help="Number of test samples (default: 20)"
    )
    args = parser.parse_args()

    DATASET_NAME = args.dataset
    SAMPLES_TRAIN = args.samples_train
    SAMPLES_TEST = args.samples_test

    # Ensure the `datasets` folder exists
    os.makedirs("./datasets", exist_ok=True)

    # Load the specified dataset
    dataset = load_dataset(DATASET_NAME)

    if hasattr(dataset, 'frame') and dataset.frame is not None:
        data = dataset.frame
    else:
        data = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
        data["target"] = dataset.target

    # Select SAMPLES_TRAIN + SAMPLES_TEST random elements from the dataset
    dataset_sample = utils.resample(data, n_samples=SAMPLES_TRAIN + SAMPLES_TEST)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        dataset_sample.iloc[:, :-1],
        dataset_sample["target"],
        train_size=SAMPLES_TRAIN,
        random_state=1
    )

    # Assemble test dataset and save it to a CSV file
    test_df = pd.DataFrame(X_test, columns=data.columns[:-1])
    test_df["target"] = y_test
    test_df.to_csv("./datasets/test_data.csv", index=False)

    # Assemble train dataset
    train_df = pd.DataFrame(X_train, columns=data.columns[:-1])
    train_df["target"] = y_train

    # Binarize the target labels
    label_binarizer = preprocessing.LabelBinarizer()
    label_binarizer_output = label_binarizer.fit_transform(train_df["target"])
    label_df = pd.DataFrame(
        label_binarizer_output,
        columns=[f"label_{cls}" for cls in label_binarizer.classes_],
        index=train_df.index
    )

    # Merge binarized labels with train dataset
    train_df_binarized = pd.concat([train_df, label_df], axis=1)

    # Save the binarized training dataset to a CSV file
    train_df_binarized.to_csv("./datasets/train_data.csv", index=False)

    # Prepare dataset for Noir
    noir_data = []
    label_columns = [f"label_{cls}" for cls in label_binarizer.classes_]
    for _, row in train_df_binarized.iterrows():
        quantized_features = quantize_row(row[data.columns[:-1]])
        labels = {label: int(row[label]) for label in label_columns}
        noir_data.append({
            "features": quantized_features,
            **labels
        })

    # Save dataset as JSON for Noir
    with open("./datasets/train_data.json", "w") as f:
        json.dump(noir_data, f, indent=4)

    print(f"Dataset '{DATASET_NAME}' prepared and saved to:")
    print("- CSV (train): ./datasets/train_data.csv")
    print("- CSV (test): ./datasets/test_data.csv")
    print("- JSON (Noir): ./datasets/train_data.json")

    # Determine the number of features and classes
    num_features = data.shape[1] - 1  # Exclude the target column
    num_classes = len(label_binarizer.classes_)

    # Save metadata
    metadata = {
        "features": num_features,
        "classes": num_classes,
        "samples_train": SAMPLES_TRAIN,
        "samples_test": SAMPLES_TEST
    }

    with open("./datasets/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata saved to ./datasets/metadata.json")

