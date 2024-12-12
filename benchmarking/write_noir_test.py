import json
import argparse

SCALE = 2**16

def format_features(features):
    return ", ".join([f"Quantized {{ x: {value} }}" for value in features])

def format_labels(labels):
    """Format labels for Noir."""
    return ",\n        ".join(
        f"[{', '.join(f'Quantized {{ x: {label} }}' for label in class_labels)}]"
        for class_labels in labels
    )

def replace_placeholders(template, inputs, labels, epochs, ratio, learning_rate):
    """
    Replace placeholders in the Noir template with actual data.
    """
    return (
        template
        .replace("// INPUTS_PLACEHOLDER", ",\n        ".join(inputs))
        .replace("// LABELS_PLACEHOLDER", format_labels(labels))
        .replace("// EPOCHS_PLACEHOLDER", str(epochs))
        .replace("// RATIO_PLACEHOLDER", str(ratio))
        .replace("// LEARNING_RATE_PLACEHOLDER", str(learning_rate))
    )

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Adjust Noir code with flexible parameters.")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs for training")
    parser.add_argument("--samples-train", type=int, default=30, help="Number of training samples")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Learning rate")
    args = parser.parse_args()

    SAMPLES_TRAIN = args.samples_train
    # Calculate ratio based on the hardcoded samples_train
    ratio = round(SCALE / SAMPLES_TRAIN)  # Compute 1/n * SCALE
    learning_rate = round(args.learning_rate * SCALE)
    
    # Load dataset
    with open("./datasets/train_data.json", "r") as f:
        dataset = json.load(f)

    # Dynamically determine the label keys (e.g., label_0, label_1, ...)
    label_keys = [key for key in dataset[0] if key.startswith("label_")]

    inputs = []
    labels = {key: [] for key in label_keys}  # Initialize label containers dynamically

    for row in dataset:
        inputs.append(f"[{format_features(row['features'])}]")
        for key in label_keys:
            labels[key].append(row[key] * SCALE)  # Scale label values

    # Convert labels to match Noir's expected format [[Quantized; N]; C]
    grouped_labels = [
        [labels[key][j] for j in range(SAMPLES_TRAIN)] for key in label_keys
    ]

    # Load the Noir template
    with open("./noir_project/src/main_template.nr", "r") as f:
        template = f.read()

    # Replace placeholders in the template
    updated_code = replace_placeholders(template, inputs, grouped_labels, args.epochs, ratio, learning_rate)

    # Write updated code to main.nr
    with open("./noir_project/src/main.nr", "w") as f:
        f.write(updated_code)

    print(f"Noir code updated for {args.epochs} epochs and saved to ./noir_project/src/main.nr")
