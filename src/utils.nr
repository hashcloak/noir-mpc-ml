pub unconstrained fn get_bytes(x: Field) -> [u8; 32] {
    x.to_be_bytes()
}

// returns x * 2^-16, interpreted as an integer
// Scaling down is needed for example after multiplying 2 Quantized values
pub fn scale_down(x: Field) -> Field {
    let lower_16_bits = x as u16;
    let result = x - lower_16_bits as Field;
    result / 65536
} 
