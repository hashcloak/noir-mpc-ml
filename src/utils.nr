// use noir_fixed_point::quantized::is_negative;

// pub unconstrained fn get_bytes(x: Field) -> [u8; 32] {
//     x.to_be_bytes()
// }

// // returns x * 2^-16, interpreted as an integer
// // Scaling down is needed for example after multiplying 2 Quantized values
// pub fn scale_down(x: Field) -> Field {
//     let mut z = x;
//     let mut bytes: [u8; 32] = [0; 32];

//     let negative = is_negative(z);

//     z = negative
//         * (
//             21888242871839275222246405745257275088548364400416034343698204186575808495616 - z + 1
//                 - z
//         )
//         + z;
    
//     unsafe {
//         //@safety to be changed
//         bytes = get_bytes(z);
//     }
//     assert(Field::from_be_bytes::<32>(bytes) == z);

//     // Division by 2^16, code as suggested by Tom French @TomAFrench
//     // Cast x to a u16 to preserve only the lowest 16 bits.
//     let lowest_16_bits = z as u16;

//     // Subtract off the lowest 16 bits so they are cleared.
//     let x_with_cleared_lower_bits = z - lowest_16_bits as Field;

//     // The lowest 16 bits are clear, `x_with_cleared_lower_bits` is divisible by `65536`,
//     // therefore field division is equivalent to integer division.
//     let mut new_x: Field = x_with_cleared_lower_bits / 65536;

//     // Flip back sign if the output is negative
//     new_x = negative
//         * (
//             21888242871839275222246405745257275088548364400416034343698204186575808495616 - new_x
//                 + 1
//                 - new_x
//         )
//         + new_x;

//     new_x
// }
