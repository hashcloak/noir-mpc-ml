import argparse
import re

def parse_noir_output(output_file, quantized_file, features, classes):
    with open(output_file, "r") as f:
        lines = f.read()

    # Extract nested Quantized values using regex
    pattern = r"Quantized \{ x: ([^}]+) \}"
    matches = re.findall(pattern, lines)

    if not matches:
        raise ValueError("No Quantized values found in Noir output.")

    # Validate the number of extracted parameters
    expected_params = classes * (features + 1)  # Each class has `features` weights + 1 bias
    if len(matches) != expected_params:
        raise ValueError(
            f"Expected {expected_params} parameters (for {classes} classes and {features} features), "
            f"but found {len(matches)}."
        )

    # Write the parameters to the quantized file
    with open(quantized_file, "w") as f:
        for param in matches:
            f.write(f"{param}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse Noir output to extract quantized parameters.")
    parser.add_argument("--output-file", required=True, help="File containing the Noir output.")
    parser.add_argument("--quantized-file", required=True, help="File to save parsed quantized parameters.")
    parser.add_argument("--features", type=int, required=True, help="Number of features in the dataset.")
    parser.add_argument("--classes", type=int, required=True, help="Number of classes in the dataset.")
    args = parser.parse_args()

    parse_noir_output(args.output_file, args.quantized_file, args.features, args.classes)
