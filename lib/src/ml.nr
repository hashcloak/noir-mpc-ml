use crate::quantized::{get_bytes, is_negative, Quantized};
use super::quantized::scale_down;

fn approx_sigmoid(x: Quantized) -> Quantized {
    // Follows https://github.com/data61/MP-SPDZ/blob/master/Compiler/ml.py#L110
    // Piece-wise approximate sigmoid as in
    // Hong et al. https://arxiv.org/abs/2002.04344
    // [-5, -2.5, 2.5, 5]
    let cuts: [Quantized; 4] = [
        Quantized::new(-327680), // -5/2^-16, -327680
        Quantized::new(-163840), // -2.5/2^-16, -163840
        Quantized::new(163840), // 163840
        Quantized::new(327680), // 327680
    ];

    let temp = Quantized::new(1819) * x;
    let outputs = [
        Quantized::new(6), // 0.0001, 0.0001 / 2^-16 = 6.5536
        temp + Quantized::new(9502), //0.02776 and 0.145, 0.02776 / 2^-16 = 1819.27936, 0.145/2^-16 = 9502.72
        (Quantized::new(11141) * x) + Quantized::new(32768), //0.17 and 0.5, 0.17 / 2^-16 = 11141.12, 0.5/2^-16 = 32768
        temp + Quantized::new(56031), //0.02776 and 0.85498, 0.85498/2^-16 = 56031.96928
        Quantized::new(65529), //0.9999 / 2^-16 = 65529.4464
    ];

    let mut res = outputs[4]; // Default to the last index in case x is above all cuts
    // Determine the correct interval index by checking against each cut
    if x <= cuts[0] {
        res = outputs[0];
    } else if x <= cuts[1] {
        res = outputs[1];
    } else if x <= cuts[2] {
        res = outputs[2];
    } else if x <= cuts[3] {
        res = outputs[3];
    }
    res
}

fn get_prediction<let M: u32>(
    weights: [Quantized; M],
    inputs: [Quantized; M],
    bias: Quantized,
) -> Quantized {
    // let z = weights dot_product inputs
    let mut z = 0;
    for i in 0..M {
        // Perform operations directly on Field elements, scale down at the end
        z += weights[i].x * inputs[i].x;
    }

    // Scale the intermediate value down due to multiplications and add bias
    approx_sigmoid(Quantized { x: scale_down(z) } + bias)
}

pub fn train<let N: u32, let M: u32>(
    epochs: u64,
    inputs: [[Quantized; M]; N],
    labels: [Quantized; N],
    learning_rate: Quantized,
    ratio: Quantized, // ratio = 1/number_samples scaled
) -> ([Quantized; M], Quantized) {
    let mut final_weights = [Quantized::zero(); M];
    let mut final_bias = Quantized::zero();

    for _ in 0..epochs {
        let mut weight_gradient = [Quantized::zero(); M];
        let mut bias_gradient = Quantized::zero();

        for j in 0..N {
            let prediction = get_prediction(final_weights, inputs[j], final_bias);
            let error = prediction - labels[j];

            // Compute gradients
            for m in 0..M {
                weight_gradient[m].x += (inputs[j][m].x * error.x);
            }
            bias_gradient += error;
        }
        // Scale down weight_gradients (due to multiplications without scaling)
        for m in 0..M {
            weight_gradient[m].x = scale_down(weight_gradient[m].x);
        }

        // Update weights and bias using the gradients
        for m in 0..M {
            final_weights[m] -= (weight_gradient[m] * learning_rate * ratio);
        }
        final_bias -= (bias_gradient * learning_rate * ratio);
    }
    (final_weights, final_bias)
}

pub fn train_multi_class<let N: u32, let M: u32, let C: u32>(
    epochs: u64,
    inputs: [[Quantized; M]; N],
    labels: [[Quantized; N]; C],
    learning_rate: Quantized,
    ratio: Quantized, // ratio = 1/number_samples scaled
) -> [([Quantized; M], Quantized); C] {
    let mut result_parameters = [([Quantized { x: 0 }; M], Quantized { x: 0 }); C];
    for i in 0..C {
        let parameters = train(epochs, inputs, labels[i], learning_rate, ratio);
        result_parameters[i] = parameters;
    }
    result_parameters
}

#[test]
fn test_training() {
    let inputs = [
        [
            Quantized { x: 334234 },
            Quantized { x: 163840 },
            Quantized { x: 196608 },
            Quantized { x: 72090 },
        ],
        [
            Quantized { x: 367002 },
            Quantized { x: 196608 },
            Quantized { x: 268698 },
            Quantized { x: 85197 },
        ],
        [
            Quantized { x: 471859 },
            Quantized { x: 209715 },
            Quantized { x: 393216 },
            Quantized { x: 117965 },
        ],
        [
            Quantized { x: 419430 },
            Quantized { x: 209715 },
            Quantized { x: 294912 },
            Quantized { x: 98304 },
        ],
        [
            Quantized { x: 281805 },
            Quantized { x: 196608 },
            Quantized { x: 72090 },
            Quantized { x: 6554 },
        ],
        [
            Quantized { x: 373555 },
            Quantized { x: 196608 },
            Quantized { x: 275251 },
            Quantized { x: 78643 },
        ],
        [
            Quantized { x: 334234 },
            Quantized { x: 242483 },
            Quantized { x: 98304 },
            Quantized { x: 26214 },
        ],
        [
            Quantized { x: 380109 },
            Quantized { x: 176947 },
            Quantized { x: 255590 },
            Quantized { x: 78643 },
        ],
        [
            Quantized { x: 439091 },
            Quantized { x: 196608 },
            Quantized { x: 327680 },
            Quantized { x: 111411 },
        ],
        [
            Quantized { x: 393216 },
            Quantized { x: 144179 },
            Quantized { x: 327680 },
            Quantized { x: 98304 },
        ],
        [
            Quantized { x: 373555 },
            Quantized { x: 190054 },
            Quantized { x: 275251 },
            Quantized { x: 85197 },
        ],
        [
            Quantized { x: 380109 },
            Quantized { x: 183501 },
            Quantized { x: 334234 },
            Quantized { x: 157286 },
        ],
        [
            Quantized { x: 334234 },
            Quantized { x: 216269 },
            Quantized { x: 111411 },
            Quantized { x: 32768 },
        ],
        [
            Quantized { x: 288358 },
            Quantized { x: 209715 },
            Quantized { x: 85197 },
            Quantized { x: 13107 },
        ],
        [
            Quantized { x: 340787 },
            Quantized { x: 268698 },
            Quantized { x: 98304 },
            Quantized { x: 6554 },
        ],
        [
            Quantized { x: 412877 },
            Quantized { x: 163840 },
            Quantized { x: 321126 },
            Quantized { x: 98304 },
        ],
        [
            Quantized { x: 439091 },
            Quantized { x: 203162 },
            Quantized { x: 367002 },
            Quantized { x: 157286 },
        ],
        [
            Quantized { x: 432538 },
            Quantized { x: 190054 },
            Quantized { x: 301466 },
            Quantized { x: 85197 },
        ],
        [
            Quantized { x: 393216 },
            Quantized { x: 144179 },
            Quantized { x: 327680 },
            Quantized { x: 98304 },
        ],
        [
            Quantized { x: 406323 },
            Quantized { x: 144179 },
            Quantized { x: 294912 },
            Quantized { x: 98304 },
        ],
        [
            Quantized { x: 439091 },
            Quantized { x: 216269 },
            Quantized { x: 373555 },
            Quantized { x: 163840 },
        ],
        [
            Quantized { x: 288358 },
            Quantized { x: 190054 },
            Quantized { x: 91750 },
            Quantized { x: 13107 },
        ],
        [
            Quantized { x: 445645 },
            Quantized { x: 209715 },
            Quantized { x: 386662 },
            Quantized { x: 150733 },
        ],
        [
            Quantized { x: 327680 },
            Quantized { x: 131072 },
            Quantized { x: 229376 },
            Quantized { x: 65536 },
        ],
        [
            Quantized { x: 327680 },
            Quantized { x: 209715 },
            Quantized { x: 78643 },
            Quantized { x: 13107 },
        ],
        [
            Quantized { x: 367002 },
            Quantized { x: 196608 },
            Quantized { x: 268698 },
            Quantized { x: 85197 },
        ],
        [
            Quantized { x: 380109 },
            Quantized { x: 176947 },
            Quantized { x: 255590 },
            Quantized { x: 78643 },
        ],
        [
            Quantized { x: 380109 },
            Quantized { x: 176947 },
            Quantized { x: 255590 },
            Quantized { x: 78643 },
        ],
        [
            Quantized { x: 399770 },
            Quantized { x: 190054 },
            Quantized { x: 308019 },
            Quantized { x: 91750 },
        ],
        [
            Quantized { x: 452198 },
            Quantized { x: 209715 },
            Quantized { x: 373555 },
            Quantized { x: 150733 },
        ],
    ];

    let labels_0 = [
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 65536 },
        Quantized { x: 0 },
        Quantized { x: 65536 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 65536 },
        Quantized { x: 65536 },
        Quantized { x: 65536 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 65536 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 65536 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 0 },
    ];

    let labels_1 = [
        Quantized { x: 65536 },
        Quantized { x: 65536 },
        Quantized { x: 0 },
        Quantized { x: 65536 },
        Quantized { x: 0 },
        Quantized { x: 65536 },
        Quantized { x: 0 },
        Quantized { x: 65536 },
        Quantized { x: 65536 },
        Quantized { x: 0 },
        Quantized { x: 65536 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 65536 },
        Quantized { x: 0 },
        Quantized { x: 65536 },
        Quantized { x: 0 },
        Quantized { x: 65536 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 65536 },
        Quantized { x: 0 },
        Quantized { x: 65536 },
        Quantized { x: 65536 },
        Quantized { x: 65536 },
        Quantized { x: 65536 },
        Quantized { x: 0 },
    ];

    let label_2 = [
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 65536 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 65536 },
        Quantized { x: 0 },
        Quantized { x: 65536 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 65536 },
        Quantized { x: 0 },
        Quantized { x: 65536 },
        Quantized { x: 0 },
        Quantized { x: 65536 },
        Quantized { x: 0 },
        Quantized { x: 65536 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 65536 },
    ];

    let epochs = 10;
    let learning_rate = Quantized::new(6554); //0.1  0.1/2^-16 6553.6
    let ratio = Quantized::new(2185); // 1/nr samples = 1/30 => (1/30)/2^-16 = 2184.53
    let parameters = train_multi_class(
        epochs,
        inputs,
        [labels_0, labels_1, label_2],
        learning_rate,
        ratio,
    );
    println(parameters);
    /*
    ========== RESULT FOR TWO CLASSES (0 AND 1) ================================
    epochs = 10
    [Quantized { x: 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593efffe01e }, Quantized { x: 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593efffb046 }, Quantized { x: 0x6ce6 }, Quantized { x: 0x2fc3 }]
    Quantized { x: 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593efffefea }
    Converted:
    Weights: [[-0.37246704]
    [-1.14930725]
    [ 1.71784973]
    [ 0.74742126]]
    Bias: -0.2356414794921875
    Comparison, Rust output for 10 epochs:
    Trained Weights: [-0.3726472362241575, -1.1493118151014219, 1.7181282744314055, 0.7479174088260542]
    Trained Bias: -0.23622522154644904
    */
}

#[test]
fn test_approx_sigmoid() {
    // Testvalues: 1.0, 0.1, -0.5, -1.1
    // Input 1.0, should return 0.67
    // Quantized values:
    // 1 / 2^-16 = 65536
    // 0.67 / 2^-16 = 43909.1
    let x = Quantized::new(65536);
    let res = approx_sigmoid(x);
    println(res); // 43909 exact answer!
    // Input 0.1, should return 0.517
    // Quantized values:
    // 0.1 / 2^-16 = 6553.6 => round to 6554
    // 0.517 / 2^-16 =  33882.112
    let x2 = Quantized::new(6554);
    let res2 = approx_sigmoid(x2);
    println(res2); // 33882
    // Input -0.5, should return 0.415
    // Quantized values:
    // -0.5 / 2^-16 = -32768
    // 0.415 / 2^-16 = 27197.44
    let x3 = Quantized::new(-32768);
    let res3 = approx_sigmoid(x3);
    println(res3); // 27198 differs by 1, due to loss of accuracy
    // Input -1.1, should return 0.31299999999999994
    // Quantized values:
    // -1.1 / 2^-16 = -72089.6 => round to -72090
    // 0.31299999999999994 / 2^-16 = 20512.7679
    let x4 = Quantized::new(-72090);
    let res4 = approx_sigmoid(x4);
    println(res4); // 20513 differs by 1, due to loss of accuracy
}
