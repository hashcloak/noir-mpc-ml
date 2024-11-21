use crate::quantized::{Quantized, get_bytes, is_negative};

fn approx_sigmoid(x: Quantized) -> Quantized {
    // Follows https://github.com/data61/MP-SPDZ/blob/master/Compiler/ml.py#L110
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

fn get_prediction(weights: [Quantized; 4], inputs: [Quantized; 4], bias: Quantized) -> Quantized {
    // let z = weights dot_product inputs
    let mut z = 0;
    for i in 0..4 {
        // Perform operations directly on Field elements, scale down at the end
        z += weights[i].x * inputs[i].x;
    }
    let mut bytes: [u8; 32] = [0; 32];
    unsafe {
        bytes = get_bytes(z);
    }
    assert(Field::from_be_bytes::<32>(bytes) == z);
    let negative = is_negative(z);

    if negative {
        z = 21888242871839275222246405745257275088548364400416034343698204186575808495616 - z + 1;
        unsafe {
            bytes = get_bytes(z);
        }

        assert(Field::from_be_bytes::<32>(bytes) == z);
    }
    // Now scale down with the same trick as in Quantized mul
    let mut truncated: [u8; 32] = [0; 32];
    for i in 0..30 {
        truncated[i + 2] = bytes[i];
    }

    let mut new_x: Field = Field::from_be_bytes::<32>(truncated);
    // Flip back sign if the output is negative
    if negative {
        new_x = 21888242871839275222246405745257275088548364400416034343698204186575808495616
            - new_x
            + 1;
    }
    approx_sigmoid(Quantized { x: new_x } + bias)
}

pub fn train(
    epochs: u64,
    inputs: [[Quantized; 4]; 30],
    labels: [Quantized],
    learning_rate: Quantized,
    ratio: Quantized, // ratio = 1/number_samples scaled
) -> ([Quantized; 4], Quantized) {
    let mut final_weights = [Quantized::zero(); 4];
    let mut final_bias = Quantized::zero();

    for _ in 0..epochs {
        let mut weight_gradient = [Quantized::zero(); 4];
        let mut bias_gradient = Quantized::zero();

        for j in 0..labels.len() {
            let prediction = get_prediction(final_weights, inputs[j], final_bias);
            let error = prediction - labels[j];

            // Compute gradients
            for m in 0..4 {
                weight_gradient[m] += (inputs[j][m] * error);
            }
            bias_gradient += error;
        }

        // Update weights and bias using the gradients
        for m in 0..4 {
            final_weights[m] -= (weight_gradient[m] * learning_rate * ratio);
        }
        final_bias -= (bias_gradient * learning_rate * ratio);
    }
    (final_weights, final_bias)
}

#[test]
fn test_training() {
    let inputs = [
        [
            Quantized { x: 360448 },
            Quantized { x: 170394 },
            Quantized { x: 288358 },
            Quantized { x: 78643 },
        ],
        [
            Quantized { x: 327680 },
            Quantized { x: 222822 },
            Quantized { x: 104858 },
            Quantized { x: 26214 },
        ],
        [
            Quantized { x: 340787 },
            Quantized { x: 268698 },
            Quantized { x: 98304 },
            Quantized { x: 6554 },
        ],
        [
            Quantized { x: 425984 },
            Quantized { x: 183501 },
            Quantized { x: 301466 },
            Quantized { x: 98304 },
        ],
        [
            Quantized { x: 367002 },
            Quantized { x: 196608 },
            Quantized { x: 294912 },
            Quantized { x: 98304 },
        ],
        [
            Quantized { x: 301466 },
            Quantized { x: 209715 },
            Quantized { x: 91750 },
            Quantized { x: 13107 },
        ],
        [
            Quantized { x: 393216 },
            Quantized { x: 144179 },
            Quantized { x: 262144 },
            Quantized { x: 65536 },
        ],
        [
            Quantized { x: 334234 },
            Quantized { x: 222822 },
            Quantized { x: 98304 },
            Quantized { x: 13107 },
        ],
        [
            Quantized { x: 432538 },
            Quantized { x: 190054 },
            Quantized { x: 301466 },
            Quantized { x: 85197 },
        ],
        [
            Quantized { x: 308019 },
            Quantized { x: 209715 },
            Quantized { x: 85197 },
            Quantized { x: 13107 },
        ],
        [
            Quantized { x: 412877 },
            Quantized { x: 216269 },
            Quantized { x: 308019 },
            Quantized { x: 104858 },
        ],
        [
            Quantized { x: 340787 },
            Quantized { x: 222822 },
            Quantized { x: 91750 },
            Quantized { x: 13107 },
        ],
        [
            Quantized { x: 380109 },
            Quantized { x: 170394 },
            Quantized { x: 262144 },
            Quantized { x: 78643 },
        ],
        [
            Quantized { x: 399770 },
            Quantized { x: 183501 },
            Quantized { x: 262144 },
            Quantized { x: 85197 },
        ],
        [
            Quantized { x: 373555 },
            Quantized { x: 183501 },
            Quantized { x: 268698 },
            Quantized { x: 85197 },
        ],
        [
            Quantized { x: 380109 },
            Quantized { x: 262144 },
            Quantized { x: 78643 },
            Quantized { x: 13107 },
        ],
        [
            Quantized { x: 314573 },
            Quantized { x: 222822 },
            Quantized { x: 124518 },
            Quantized { x: 13107 },
        ],
        [
            Quantized { x: 367002 },
            Quantized { x: 196608 },
            Quantized { x: 294912 },
            Quantized { x: 98304 },
        ],
        [
            Quantized { x: 321126 },
            Quantized { x: 196608 },
            Quantized { x: 91750 },
            Quantized { x: 13107 },
        ],
        [
            Quantized { x: 373555 },
            Quantized { x: 249037 },
            Quantized { x: 111411 },
            Quantized { x: 19661 },
        ],
        [
            Quantized { x: 327680 },
            Quantized { x: 216269 },
            Quantized { x: 91750 },
            Quantized { x: 13107 },
        ],
        [
            Quantized { x: 406323 },
            Quantized { x: 190054 },
            Quantized { x: 281805 },
            Quantized { x: 85197 },
        ],
        [
            Quantized { x: 334234 },
            Quantized { x: 249037 },
            Quantized { x: 124518 },
            Quantized { x: 26214 },
        ],
        [
            Quantized { x: 314573 },
            Quantized { x: 196608 },
            Quantized { x: 91750 },
            Quantized { x: 6554 },
        ],
        [
            Quantized { x: 327680 },
            Quantized { x: 196608 },
            Quantized { x: 104858 },
            Quantized { x: 13107 },
        ],
        [
            Quantized { x: 288358 },
            Quantized { x: 209715 },
            Quantized { x: 85197 },
            Quantized { x: 13107 },
        ],
        [
            Quantized { x: 301466 },
            Quantized { x: 209715 },
            Quantized { x: 91750 },
            Quantized { x: 13107 },
        ],
        [
            Quantized { x: 294912 },
            Quantized { x: 150733 },
            Quantized { x: 85197 },
            Quantized { x: 19661 },
        ],
        [
            Quantized { x: 334234 },
            Quantized { x: 249037 },
            Quantized { x: 124518 },
            Quantized { x: 26214 },
        ],
        [
            Quantized { x: 334234 },
            Quantized { x: 229376 },
            Quantized { x: 91750 },
            Quantized { x: 19661 },
        ],
    ];

    let labels = [
        Quantized { x: 1 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 1 },
        Quantized { x: 1 },
        Quantized { x: 0 },
        Quantized { x: 1 },
        Quantized { x: 0 },
        Quantized { x: 1 },
        Quantized { x: 0 },
        Quantized { x: 1 },
        Quantized { x: 0 },
        Quantized { x: 1 },
        Quantized { x: 1 },
        Quantized { x: 1 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 1 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 1 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 0 },
        Quantized { x: 0 },
    ];

    let epochs = 10;
    let learning_rate = Quantized::new(6554); //0.1  0.1/2^-16 6553.6
    let ratio = Quantized::new(2185); // 1/nr samples = 1/30 => (1/30)/2^-16 = 2184.53
    let (final_weights, final_bias) = train(epochs, inputs, labels, learning_rate, ratio);
    println(final_weights);
    println(final_bias);
    // p = 21888242871839275222246405745257275088548364400416034343698204186575808495617
    /*
    epoch = 10
    [Quantized { x: 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593efff798a }, Quantized { x: 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593efffa3dd }, Quantized { x: 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593efffd87e }, Quantized { x: 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593effff980 }]
    Quantized { x: 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593efffe51d }
    weights
    21888242871839275222246405745257275088548364400416034343698204186575808461194
    21888242871839275222246405745257275088548364400416034343698204186575808472029
    21888242871839275222246405745257275088548364400416034343698204186575808485502
    21888242871839275222246405745257275088548364400416034343698204186575808493952
    converted:
    (21888242871839275222246405745257275088548364400416034343698204186575808495617- 21888242871839275222246405745257275088548364400416034343698204186575808461194) / 2^16
    -0.5252532958984375
    (21888242871839275222246405745257275088548364400416034343698204186575808495617-21888242871839275222246405745257275088548364400416034343698204186575808472029) / 2^16
    -0.35992431640625
    (21888242871839275222246405745257275088548364400416034343698204186575808495617-21888242871839275222246405745257275088548364400416034343698204186575808485502) / 2^16
    -0.1543426513671875
    (21888242871839275222246405745257275088548364400416034343698204186575808495617-21888242871839275222246405745257275088548364400416034343698204186575808493952) / 2^16
    -0.0254058837890625
    [-0.5252532958984375, -0.35992431640625, -0.1543426513671875, -0.0254058837890625]
    In Rust for same samples and #epochs
    [-0.5252033301021997, -0.3599251494270883, -0.15437820313086892, -0.02547627906962113]
    Bias Noir
    21888242871839275222246405745257275088548364400416034343698204186575808488733
    converted:
    (21888242871839275222246405745257275088548364400416034343698204186575808495617- 21888242871839275222246405745257275088548364400416034343698204186575808488733) / 2^16
    -0.10504150390625
    Rust: -0.1050845195834892
    */
    /*
    epoch = 100
    [Quantized { x: 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593efff4ecb }, Quantized { x: 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593efff86e6 }, Quantized { x: 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593efffcbf3 }, Quantized { x: 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593effff7ee }]
    Quantized { x: 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593efffdc2c }
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
