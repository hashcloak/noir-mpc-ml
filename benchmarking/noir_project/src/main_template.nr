use noir_mpc_ml::ml::{train, train_multi_class};
use noir_mpc_ml::quantized::Quantized;

fn main(x: Field, y: pub Field) {
    assert(x != y);
}

#[test]
fn test_train() {
    let inputs = [
        // INPUTS_PLACEHOLDER
    ];

    let labels = [
        // LABELS_PLACEHOLDER
    ];

    let epochs = // EPOCHS_PLACEHOLDER;

    let learning_rate = Quantized::new(
        // LEARNING_RATE_PLACEHOLDER
    );
    let ratio = Quantized::new(
        // RATIO_PLACEHOLDER
    );
    let parameters = train_multi_class(
        epochs,
        inputs,
        labels,
        learning_rate,
        ratio,
    );
    println(parameters);
}
