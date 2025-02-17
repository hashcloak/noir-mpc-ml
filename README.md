# Publicly Verifiable & Private Collaborative ML Model Training

This repository contains the implementation of private collaborative and public
verifiable training for the logistic regression model using Noir. This project
comes as a result of the [NGR Request for Private Shared States using Noir](https://github.com/orgs/noir-lang/discussions/6317)
sponsored by Aztec Labs.

The project contains the following features implemented in the Noir programming
language:

- An implementation of fixed-point numbers using quantized arithmetic.
- An implementation of *deterministic* logistic regression training for both two
classes and multiple classes.
- Benchmarking and test of the performance and accuracy of the implementation
using the Iris plants dataset and the Breast cancer dataset.

## How to use

First, you need to include the library in your `Nargo.toml` file as follows:

```toml
[package]
name = "noir_project"
type = "bin"
authors = [""]
compiler_version = ">=0.36.0"

[dependencies]
noir_mpc_ml = { git = "https://github.com/hashcloak/noir-mpc-ml/tree/benchmarking/lib", branch = "master" }
```

Bellow, we present an example of how to use the training for a dataset with 30
rows, 4 features, and 3 classes. For this example, suppose that the Prover,
wants to prove that he has the dataset that produces a certain set of parameters
known to the verifier for a logistic regression model using a public number of
epochs and learning rate. Hence the source code will be as follows:

```rust
use noir_mpc_ml::ml::train_multi_class;
use noir_mpc_ml::quantized::Quantized;

fn main(
    data: [[Quantized; 4]; 30],
    labels: [[Quantized; 30]; 3],
    learning_rate: pub Quantized,
    ratio: pub Quantized,
    epochs: pub u64,
    parameters: [([Quantized; 4], Quantized); 3],
) {
    let parameters_train = train_multi_class(epochs, data, labels, learning_rate, ratio);
    assert(parameters == parameters_train);
}
```

Some of the concepts present in the previous example will be explained in dept
later, but we will explain some basic concepts here. The `Quantized` type
represents a fixed-point number. To train a model, we use the `train_multiclass`
method which receives the features of each sample, the labels, the ratio which
is $1 / N$ where $N$ is the number of samples, and the number of epochs for the
training.

In this case, notice that the labels are provided in a $N \times C$ matrix where
$N$ is the number of samples and $C$ is the number of classes. The `labels`
matrix will have a 1 in the position $(i, c)$ if the $i$-th sample is of class
$c$, otherwise, the entry will have a 0.

## Benchmarks

We executed benchmarks for the logistic regression library using the Iris and the Wine datasets. To execute these benchmarks yourself, you can go to the `benchmarks/` folder in which you will find instruction to execute and test the library.

### Number of gates

In the following tables, we present the number of gates for different epochs and number of training samples using the Iris and the Wine dataset. The number of gates is measured using the Noir tooling.

#### For the Iris dataset

| **Epochs** | **# of train samples** | **ACIR opcodes** | **# of gates** |
|:----------:|------------------------|------------------|----------------|
|         10 |                     20 |          231,396 |        402,403 |
|         10 |                     60 |          618,684 |      1,079,991 |
|         20 |                     20 |          484,164 |        844,150 |
|         20 |                     60 |        1,299,372 |      2,281,998 |
|         30 |                     20 |          738,135 |      1,287,658 |
|         30 |                     60 |        1,982,583 |      3,488,406 |

#### For the Wine dataset

| **Epochs** | **# of train samples** | **ACIR opcodes** | **# of gates** |
|:----------:|------------------------|------------------|----------------|
|         10 |                     20 |          314,556 |        535,513 |
|         10 |                     60 |          729,762 |      1,250,649 |
|         20 |                     20 |          650,994 |      1,110,790 |
|         20 |                     60 |        1,523,220 |      2,625,636 |
|         30 |                     20 |          988,635 |      1,687,828 |

### Training using co-noir

The following table shows the training time using co-noir for the Iris dataset using a server with an AMD EPYC Processor and 32 GB of RAM.

| Epochs | # of train samples | Training time [sec] |
|--------|--------------------|----------------------|
|     10 |                 30 |                1,845 |
|     10 |                 50 |                2,878 |
|     20 |                 30 |                4,557 |
|     20 |                 50 |                6,875 |

## Fixed-point arithmetic

The fixed point arithmetic follows the strategy presented in the paper of
[Catrina and Saxena](https://www.ifca.ai/pub/fc10/31_47.pdf). In the paper, the
authors propose a way to represent fixed-point numbers using field elements for
MPC protocols. Additionally, the propose MPC protocols for addition,
multiplication and division. In the context of zero-knowledge proofs, we saw this
paper as an opportunity to implement the fixed-point arithmetic given that the
primitive data type in Noir is the `Field`. This allows us to implement the
fixed-point data type without relying on native integer types from Noir, whose
impose an additional overhead to the computation.

In the representation, the fixed-point numbers are represented by a `Field`
element that is wrapped in the `Quantized` struct. This field element will
represent a fractional number that has $k$ bits in total and $f$ of those $k$
bits are used to represent the decimal part. We will denote this set of
fractional numbers as $\mathbb{Q}\_{\langle k, f \rangle}$. An element
$\tilde{x} \in \mathbb{Q}\_{\langle k, f \rangle}$ can be encoded as a `Field`
element by computing $(x = \tilde{x} \cdot 2^{-f}) \mod p$ where $p$ is the order
of the `Field`. Adding two `Quantized` elements corresponds to add both encodings.
However, the multiplication requires a truncation given that multiplying both encodings
results in a number with precision $2f$.

## Logistic regression training

The implementation of the logistic training algorithm is done by using the
gradient descent method. This algorithm is an iterative method that updates the
weight of the parameters of the model in the direction of the gradient of a
log-loss function.

The algorithm is as follows:

**Inputs:** the data samples $X \in \mathbb{R}^{(n \times m)}$, the labels
$y \in \mathbb{R}^{n}$, the learning rate $\alpha \in \mathbb{R}$, and the
number of epochs $E$.

1. Let $w \in \mathbb{R}^m$ and $b \in \mathbb{R}$ initialized in zero.
2. Execute the following steps $E$ times:

- For each $j \in {1, \dots, m}$, $w_j = w_j - (\alpha / n) \cdot \sum_{i=1}^n [\sigma(w \cdot x_i + b) - y_i] \cdot X_{ij}$
- $b = b - (\alpha / n) \cdot \sum_{i=1}^n [\sigma(w \cdot x_i + b) - y_i]$

3. Return $w$ and $b$.

## Acknowledgements

We thank Aztec for funding this project.