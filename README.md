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
fractional numbers as $\mathbb{Q}_{\langle k, f \rangle}$.

## Logistic regression training

## References
