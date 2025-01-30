// use super::utils::scale_down;
use crate::quantized::Quantized;

fn approx_sigmoid(x: Quantized) -> Quantized {
    // Follows https://github.com/data61/MP-SPDZ/blob/master/Compiler/ml.py#L110
    // Piece-wise approximate sigmoid as in
    // Hong et al. https://arxiv.org/abs/2002.04344
    // [-5, -2.5, 2.5, 5]
    let cuts: [Quantized; 4] = [
        Quantized { x: U128::from_u64s_le(327680 as u64,0), neg: true, }, // -5/2^-16, -327680
        Quantized { x: U128::from_u64s_le(163840 as u64,0), neg: true, }, // -2.5/2^-16, -163840
        Quantized { x: U128::from_u64s_le(163840 as u64,0), neg: false}, // 163840
        Quantized { x: U128::from_u64s_le(327680 as u64,0), neg: false}, // 327680
    ];

    let temp = Quantized{ x: U128::from_u64s_le(1819 as u64,0), neg: false } * x;
    let outputs = [
        Quantized { x: U128::from_u64s_le(6 as u64,0), neg: false }, // 0.0001, 0.0001 / 2^-16 = 6.5536
        temp + Quantized { x: U128::from_u64s_le(9502 as u64,0), neg: false }, // 0.02776 and 0.145, 0.02776 / 2^-16 = 1819.27936, 0.145/2^-16 = 9502.72
        (Quantized { x: U128::from_u64s_le(11141 as u64,0), neg: false } * x) + Quantized { x: U128::from_u64s_le(32768 as u64,0),neg: false }, // 0.17 and 0.5, 0.17 / 2^-16 = 11141.12, 0.5/2^-16 = 32768
        temp + Quantized { x: U128::from_u64s_le(56031 as u64,0), neg: false }, // 0.02776 and 0.85498, 0.85498/2^-16 = 56031.96928
        Quantized { x: U128::from_u64s_le(65529 as u64,0), neg: false }, // 0.9999 / 2^-16 = 65529.4464
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
    let mut z = Quantized::zero();
    for i in 0..M {
        z += weights[i] * inputs[i];
    }

    approx_sigmoid( z + bias)
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
                weight_gradient[m] += (inputs[j][m] * error);
            }
            bias_gradient += error;
        }
        // // Scale down weight_gradients (due to multiplications without scaling)
        // for m in 0..M {
        //     weight_gradient[m].x = scale_down(weight_gradient[m].x);
        // }

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
    let mut result_parameters = [([Quantized::zero(); M], Quantized::zero() ); C];
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
            Quantized::new_pos(334234),
            Quantized::new_pos(163840),
            Quantized::new_pos(196608),
            Quantized::new_pos(72090),
        ],
        [
            Quantized::new_pos(367002),
            Quantized::new_pos(196608),
            Quantized::new_pos(268698),
            Quantized::new_pos(85197),
        ],
        [
            Quantized::new_pos(471859),
            Quantized::new_pos(209715),
            Quantized::new_pos(393216),
            Quantized::new_pos(117965),
        ],
        [
            Quantized::new_pos(419430),
            Quantized::new_pos(209715),
            Quantized::new_pos(294912),
            Quantized::new_pos(98304),
        ],
        [
            Quantized::new_pos(281805),
            Quantized::new_pos(196608),
            Quantized::new_pos(72090),
            Quantized::new_pos(6554),
        ],
        [
            Quantized::new_pos(373555),
            Quantized::new_pos(196608),
            Quantized::new_pos(275251),
            Quantized::new_pos(78643),
        ],
        [
            Quantized::new_pos(334234),
            Quantized::new_pos(242483),
            Quantized::new_pos(98304),
            Quantized::new_pos(26214),
        ],
        [
            Quantized::new_pos(380109),
            Quantized::new_pos(176947),
            Quantized::new_pos(255590),
            Quantized::new_pos(78643),
        ],
        [
            Quantized::new_pos(439091),
            Quantized::new_pos(196608),
            Quantized::new_pos(327680),
            Quantized::new_pos(111411),
        ],
        [
            Quantized::new_pos(393216),
            Quantized::new_pos(144179),
            Quantized::new_pos(327680),
            Quantized::new_pos(98304),
        ],
        [
            Quantized::new_pos(373555),
            Quantized::new_pos(190054),
            Quantized::new_pos(275251),
            Quantized::new_pos(85197),
        ],
        [
            Quantized::new_pos(380109),
            Quantized::new_pos(183501),
            Quantized::new_pos(334234),
            Quantized::new_pos(157286),
        ],
        [
            Quantized::new_pos(334234),
            Quantized::new_pos(216269),
            Quantized::new_pos(111411),
            Quantized::new_pos(32768),
        ],
        [
            Quantized::new_pos(288358),
            Quantized::new_pos(209715),
            Quantized::new_pos(85197),
            Quantized::new_pos(13107),
        ],
        [
            Quantized::new_pos(340787),
            Quantized::new_pos(268698),
            Quantized::new_pos(98304),
            Quantized::new_pos(6554),
        ],
        [
            Quantized::new_pos(412877),
            Quantized::new_pos(163840),
            Quantized::new_pos(321126),
            Quantized::new_pos(98304),
        ],
        [
            Quantized::new_pos(439091),
            Quantized::new_pos(203162),
            Quantized::new_pos(367002),
            Quantized::new_pos(157286),
        ],
        [
            Quantized::new_pos(432538),
            Quantized::new_pos(190054),
            Quantized::new_pos(301466),
            Quantized::new_pos(85197),
        ],
        [
            Quantized::new_pos(393216),
            Quantized::new_pos(144179),
            Quantized::new_pos(327680),
            Quantized::new_pos(98304),
        ],
        [
            Quantized::new_pos(406323),
            Quantized::new_pos(144179),
            Quantized::new_pos(294912),
            Quantized::new_pos(98304),
        ],
        [
            Quantized::new_pos(439091),
            Quantized::new_pos(216269),
            Quantized::new_pos(373555),
            Quantized::new_pos(163840),
        ],
        [
            Quantized::new_pos(288358),
            Quantized::new_pos(190054),
            Quantized::new_pos(91750),
            Quantized::new_pos(13107),
        ],
        [
            Quantized::new_pos(445645),
            Quantized::new_pos(209715),
            Quantized::new_pos(386662),
            Quantized::new_pos(150733),
        ],
        [
            Quantized::new_pos(327680),
            Quantized::new_pos(131072),
            Quantized::new_pos(229376),
            Quantized::new_pos(65536),
        ],
        [
            Quantized::new_pos(327680),
            Quantized::new_pos(209715),
            Quantized::new_pos(78643),
            Quantized::new_pos(13107),
        ],
    [
        Quantized::new_pos(367002),
        Quantized::new_pos(196608),
        Quantized::new_pos(268698),
        Quantized::new_pos(85197),
    ],
    [
        Quantized::new_pos(380109),
        Quantized::new_pos(176947),
        Quantized::new_pos(255590),
        Quantized::new_pos(78643),
    ],
    [
        Quantized::new_pos(380109),
        Quantized::new_pos(176947),
        Quantized::new_pos(255590),
        Quantized::new_pos(78643),
    ],
    [
        Quantized::new_pos(399770),
        Quantized::new_pos(190054),
        Quantized::new_pos(308019),
        Quantized::new_pos(91750),
    ],
    [
        Quantized::new_pos(452198),
        Quantized::new_pos(209715),
        Quantized::new_pos(373555),
        Quantized::new_pos(150733),
    ],
];

let labels_0 = [
    Quantized::new_pos(0),
    Quantized::new_pos(0),
    Quantized::new_pos(0),
    Quantized::new_pos(0),
    Quantized::new_pos(65536),
    Quantized::new_pos(0),
    Quantized::new_pos(65536),
    Quantized::new_pos(0),
    Quantized::new_pos(0),
    Quantized::new_pos(0),
    Quantized::new_pos(0),
    Quantized::new_pos(0),
    Quantized::new_pos(65536),
    Quantized::new_pos(65536),
    Quantized::new_pos(65536),
    Quantized::new_pos(0),
    Quantized::new_pos(0),
    Quantized::new_pos(0),
    Quantized::new_pos(0),
    Quantized::new_pos(0),
    Quantized::new_pos(0),
    Quantized::new_pos(65536),
    Quantized::new_pos(0),
    Quantized::new_pos(0),
    Quantized::new_pos(65536),
    Quantized::new_pos(0),
    Quantized::new_pos(0),
    Quantized::new_pos(0),
    Quantized::new_pos(0),
    Quantized::new_pos(0),
];

let labels_1 = [
    Quantized::new_pos(65536),
    Quantized::new_pos(65536),
    Quantized::new_pos(0),
    Quantized::new_pos(65536),
    Quantized::new_pos(0),
    Quantized::new_pos(65536),
    Quantized::new_pos(0),
    Quantized::new_pos(65536),
    Quantized::new_pos(65536),
    Quantized::new_pos(0),
    Quantized::new_pos(65536),
    Quantized::new_pos(0),
    Quantized::new_pos(0),
    Quantized::new_pos(0),
    Quantized::new_pos(0),
    Quantized::new_pos(65536),
    Quantized::new_pos(0),
    Quantized::new_pos(65536),
    Quantized::new_pos(0),
    Quantized::new_pos(65536),
    Quantized::new_pos(0),
    Quantized::new_pos(0),
    Quantized::new_pos(0),
    Quantized::new_pos(65536),
    Quantized::new_pos(0),
    Quantized::new_pos(65536),
    Quantized::new_pos(65536),
    Quantized::new_pos(65536),
    Quantized::new_pos(65536),
    Quantized::new_pos(0),
];

let label_2 = [
    Quantized::new_pos(0),
    Quantized::new_pos(0),
    Quantized::new_pos(65536),
    Quantized::new_pos(0),
    Quantized::new_pos(0),
    Quantized::new_pos(0),
    Quantized::new_pos(0),
    Quantized::new_pos(0),
    Quantized::new_pos(0),
    Quantized::new_pos(65536),
    Quantized::new_pos(0),
    Quantized::new_pos(65536),
    Quantized::new_pos(0),
    Quantized::new_pos(0),
    Quantized::new_pos(0),
    Quantized::new_pos(0),
    Quantized::new_pos(65536),
    Quantized::new_pos(0),
    Quantized::new_pos(65536),
    Quantized::new_pos(0),
    Quantized::new_pos(65536),
    Quantized::new_pos(0),
    Quantized::new_pos(65536),
    Quantized::new_pos(0),
    Quantized::new_pos(0),
    Quantized::new_pos(0),
    Quantized::new_pos(0),
    Quantized::new_pos(0),
    Quantized::new_pos(0),
    Quantized::new_pos(65536),
];


    let epochs = 10;
    let learning_rate = Quantized::new_pos(6554); //0.1  0.1/2^-16 6553.6
    let ratio = Quantized::new_pos(2185); // 1/nr samples = 1/30 => (1/30)/2^-16 = 2184.53
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
    [Quantized { x: 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593efffe01e, neg: false }, Quantized { x: 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593efffb046, neg: false }, Quantized { x: 0x6ce6, neg: false }, Quantized { x: 0x2fc3 }]
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
    // Test values: 1.0, 0.1, -0.5, -1.1
    // Input 1.0, should return 0.67
    // Quantized values:
    // 1 / 2^-16 = 65536
    // 0.67 / 2^-16 = 43909.1
    let x = Quantized::new_pos(65536);
    let res = approx_sigmoid(x);
    println(res); // 43909 exact answer!

    // Input 0.1, should return 0.517
    // Quantized values:
    // 0.1 / 2^-16 = 6553.6 => round to 6554
    // 0.517 / 2^-16 =  33882.112
    let x2 = Quantized::new_pos(6554);
    let res2 = approx_sigmoid(x2);
    println(res2); // 33882

    // Input -0.5, should return 0.415
    // Quantized values:
    // -0.5 / 2^-16 = -32768
    // 0.415 / 2^-16 = 27197.44
    let x3 = Quantized::new_neg(32768);
    let res3 = approx_sigmoid(x3);
    println(res3); // 27198 differs by 1, due to loss of accuracy

    // Input -1.1, should return 0.31299999999999994
    // Quantized values:
    // -1.1 / 2^-16 = -72089.6 => round to -72090
    // 0.31299999999999994 / 2^-16 = 20512.7679
    let x4 = Quantized::new_neg(72090);
    let res4 = approx_sigmoid(x4);
    println(res4); // 20513 differs by 1, due to loss of accuracy
}
