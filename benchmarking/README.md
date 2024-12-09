# Benchmarking for ML lib

The current script does the following:
1. Takes train & test data from a dataset in sklearn
2. Fills a Noir test with the training data that calls `train_multi_class`
3. Runs Noir test and saves printed weights & bias
4. Calculates accuracy of those parameters according to the test data

Note that there are conversions happening of the data in between because the original data is in floats and the Noir library works with Quantized. 

## Run

### Single test

```bash
$ ./run_single_test.sh
```

Make executionable with `$ chmod +x run_single_test.sh`. 

Optionally, define the following parameters
- `--epochs=`
- `--samples-train=`
- `--samples-test=`
- `--dataset=` available options: "iris, wine, digits, diabetes, linnerud"

Default values:
- epochs 10
- train samples 30
- test samples 20
- dataset "iris"

#### Examples

Dataset [digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html), 20 epochs, 50 training samples and 20 (default) test samples. 
```bash
$ ./cleanup.sh
$ ./run_single_test.sh --dataset=digits --epochs=20 --samples-train=50
```

### Prove & verify

This generates data like above and populates a `main` function that takes as input all the features information and calls `train_multi_class`. Then, it follows [these steps](https://noir-lang.org/docs/getting_started/quick_start#compiling-and-executing) to execute the program, prove and verify the proof. 

```bash
$ ./cleanup.sh
$ ./run_proof_example.sh
```

### Obtain gatecount

This creates the same Noir program as for proving & verifying but instead of executing those steps, just obtains the gatecount. 

For a single set of parameters:
```bash
$ ./cleanup.sh
$ ./run_get_gatecount.sh
```

Running multiple benchmarks at once, gathering output in `benches/benchmarks.csv`:
```bash
$ ./cleanup.sh
$ ./generate_gatecounts_csv.sh
```