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

