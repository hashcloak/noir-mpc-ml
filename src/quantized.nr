use std::ops::{Add, Sub, Mul};
use std::cmp::Ordering;
use std::field::bn254::decompose;

// Scale 2^-16 = 0.0000152587890625 (divide by 65536)
// p = 21888242871839275222246405745257275088548364400416034343698204186575808495617

pub struct Quantized {
    // if value positive: x*scale
    // if negative: (p-x)*scale, where p is the prime of native Field (in this case from barretenberg)
    // the original value is max 60 bits, positive or negative,
    // this leaves enough space for additions and additions of single multiplications
    // 2^60-1 = 1152921504606846975
    // -1152921504606846975 <= x <= 1152921504606846975
    // -1152921504606846975/65536 <= value <= 1152921504606846975/65536
    pub x: Field,
}

pub unconstrained fn get_bytes(x: Field) -> [u8; 32] {
    x.to_be_bytes()
}

impl Quantized {

    pub fn zero() -> Self {
        Quantized { x: 0 }
    }

    pub fn new(x: Field) -> Self {
        Self { x: x }
    }

    fn add(self: Self, other: Self) -> Self {
        Quantized { x: self.x + other.x } // if one is negative, this wraps around automatically
    }

    fn sub(self: Self, other: Self) -> Self {
        Quantized { x: self.x - other.x }
    }

    fn mul(self: Self, other: Self) -> Self {
        let mut temp: Field = self.x * other.x;
        let mut bytes: [u8; 32] = [0; 32];
        unsafe {
            bytes = get_bytes(temp);
        }

        assert(Field::from_be_bytes::<32>(bytes) == temp);

        let negative = is_negative(temp);

        // To correctly scale down we temporary flip the sign
        if negative == 1 {
            temp = 21888242871839275222246405745257275088548364400416034343698204186575808495616
                - temp
                + 1;
            unsafe {
                bytes = get_bytes(temp);
            }

            assert(Field::from_be_bytes::<32>(bytes) == temp);
        }

        // Chop off 2 bytes to scale down by 2^16 and truncate
        let mut truncated: [u8; 32] = [0; 32];
        for i in 0..30 {
            // [0, 0, am, .., a2] (chop off a0 and a1)
            truncated[i + 2] = bytes[i];
        }

        let mut new_x: Field = Field::from_be_bytes::<32>(truncated);
        // Flip back sign if the output is negative
        if negative == 1 {
            new_x = 21888242871839275222246405745257275088548364400416034343698204186575808495616
                - new_x
                + 1;
        }

        Quantized { x: new_x }
    }
}

impl Add for Quantized {
    fn add(self, other: Self) -> Self {
        self.add(other)
    }
}

impl Sub for Quantized {
    fn sub(self, other: Self) -> Self {
        self.sub(other)
    }
}

impl Mul for Quantized {
    fn mul(self, other: Self) -> Self {
        self.mul(other)
    }
}

// check if there are bytes from 0-15, then it is negative
// This assumes the values that are multiplied have max ~60-63 bits
// returns 1 for true, 0 for false
pub fn is_negative(x: Field) -> Field {
    let (_, higher_bytes) = decompose(x);
    if higher_bytes == 0 {
        0
    } else {
        1
    }
}

impl Ord for Quantized {
    fn cmp(self: Self, other: Self) -> Ordering {
        if self.x == other.x {
            Ordering::equal()
        }

        let (_, sub_hi) = decompose(self.x - other.x);
        if (sub_hi == 0) {
            Ordering::greater()
        } else {
            Ordering::less()
        }
    }
}

fn main(x: Field, y: pub Field) {
    assert(x != y);
}

#[test]
fn test_order() {
    // Test 1: comparison between positive and negative value.
    // a = 0.2 and b = -0.2
    let a: Field = 13107;
    let a_quantized = Quantized { x: a };
    let b = 21888242871839275222246405745257275088548364400416034343698204186575808482510;
    let b_quantized = Quantized { x: b };
    assert(a_quantized > b_quantized);

    // Test 2: comparison between two possitive numbers.
    // a = 1 and b = 0.2.
    // Then a * 2^16 = 65536
    let a: Field = 65536;
    let a_quantized = Quantized { x: a };
    let b = 13107;
    let b_quantized = Quantized { x: b };
    assert(a_quantized > b_quantized);

    // Test 3: comparison between two possitive numbers where the roles are
    // inverted.
    // a = 1 and b = 1.2.
    // Then a * 2^16 = 65536
    let a: Field = 65536;
    let a_quantized = Quantized { x: a };
    let b = 78643;
    let b_quantized = Quantized { x: b };
    assert(a_quantized < b_quantized);

    // Test 4: comparison between two negative numbers.
    // a = -1 and b = -0.2.
    // Then a * 2^16 = -65536 therefore taking mod p we obtain that
    // a = 21888242871839275222246405745257275088548364400416034343698204186575808430081
    // b = 21888242871839275222246405745257275088548364400416034343698204186575808482510
    let a: Field = 21888242871839275222246405745257275088548364400416034343698204186575808430081;
    let a_quantized = Quantized { x: a };
    let b = 21888242871839275222246405745257275088548364400416034343698204186575808482510;
    let b_quantized = Quantized { x: b };
    assert(a_quantized < b_quantized);

    // 21888242871839275222246405745257275088548364400416034343698204186575808495617 - 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593efff8001
    // - 32768
    // 21888242871839275222246405745257275088548364400416034343698204186575808495617 - 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593effb0001
    let a = Quantized { x: 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593efff8001 };
    // - 327680
    let b = Quantized { x: 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593effb0001 };
    assert(a > b);
}

#[test]
fn test_is_negative() {
    // Test 1: checking if it's negative for a positive value.
    // a = 0.2   =>   0.2 * 2^16 = 13107
    let a: Field = 13107;
    assert(is_negative(a) == 0);

    // Test 2: checking if it's negative for a negative value.
    // a = -0.2  =>   -0.2 * 2^16 = -13107
    // Hence, -13107 mod p = p - 13107 = 21888242871839275222246405745257275088548364400416034343698204186575808482510
    let a = 21888242871839275222246405745257275088548364400416034343698204186575808482510;
    assert(is_negative(a) == 1);

    // Test 3: checing if it's negative for a negative larger value.
    // a = -1301.34  =>  -1301.34 * 2^16 = -85284618
    // Hence, -85284618 mod p = p - 85284618 = 21888242871839275222246405745257275088548364400416034343698204186575723210999
    let a = 21888242871839275222246405745257275088548364400416034343698204186575723210999;
    assert(is_negative(a) == 1);
}

#[test]
fn test_add() {
    // Max value 2^60-1, positive and negative
    let a = 1152921504606846975;
    let b = -1152921504606846975;
    let a_quantized = Quantized { x: a };
    let b_quantized = Quantized { x: b };
    let addition_quantized = a_quantized + b_quantized;
    assert(addition_quantized.x == 0);

    // Test case 1: Max value 2^60 - 1, positive and zero
    let a1 = 1152921504606846975;
    let b1 = 0;
    let a1_quantized = Quantized { x: a1 };
    let b1_quantized = Quantized { x: b1 };
    let addition1_quantized = a1_quantized + b1_quantized;
    assert(addition1_quantized.x == a1);

    // Test case 2: Small positive values
    let a2 = 12345;
    let b2 = 67890;
    let a2_quantized = Quantized { x: a2 };
    let b2_quantized = Quantized { x: b2 };
    let addition2_quantized = a2_quantized + b2_quantized;
    assert(addition2_quantized.x == (a2 + b2));

    // Test case 3: Positive and negative values resulting in a non-zero positive result
    let a3 = 50000;
    let b3 = -30000;
    let a3_quantized = Quantized { x: a3 };
    let b3_quantized = Quantized { x: b3 };
    let addition3_quantized = a3_quantized + b3_quantized;
    assert(addition3_quantized.x == 20000);

    // Test case 4: Positive and negative values resulting in a non-zero negative result
    let a4 = 30000;
    let b4 = -50000;
    let a4_quantized = Quantized { x: a4 };
    let b4_quantized = Quantized { x: b4 };
    let addition4_quantized = a4_quantized + b4_quantized;
    assert(addition4_quantized.x == -20000);

    // Test case 5: Add two negative values resulting in a negative value
    let a5 = -40000;
    let b5 = -20000;
    let a5_quantized = Quantized { x: a5 };
    let b5_quantized = Quantized { x: b5 };
    let addition5_quantized = a5_quantized + b5_quantized;
    assert(addition5_quantized.x == -60000);
}

#[test]
fn test_mul() {
    // Test case 1: Small positive values
    let a1 = 12345; // original value 12345/2^16 = 0.1883697509765625
    let b1 = 67890; // original value 67890/2^16 = 1.035919189453125
    let a1_quantized = Quantized { x: a1 };
    let b1_quantized = Quantized { x: b1 };
    // new value = 0.1883697509765625 * 1.035919189453125 = 0.1951358397491276264190673828125
    // x = 0.1951358397491276264190673828125 * 2^16 = 12788.422393798828125 => 12788
    let mult1_quantized = a1_quantized * b1_quantized;
    assert(mult1_quantized.x == 12788);

    // Test case 2: Large positive values
    let a2 = 1152921504606846975; // Max value 2^60 - 1, original 1152921504606846975/2^16 = 17592186044415.9999847412109375
    let b2 = 2; // original 2/2^16 = 0.000030517578125
    let a2_quantized = Quantized { x: a2 };
    let b2_quantized = Quantized { x: b2 };
    let mult2_quantized = a2_quantized * b2_quantized;
    // 17592186044415.9999847412109375 * 0.000030517578125 = 536870911.9999999995343387126922607421875
    // x/2^16 = value;  mult by 2^16 and truncate to whole number => x = 35184372088831
    assert(mult2_quantized.x == 35184372088831);

    // Test case 3: Large positive and small positive value
    let a3 = 1152921504606846975; // Max value 2^60 - 1, original 1152921504606846975/2^16 = 17592186044415.9999847412109375
    let b3 = 1; // original value 1/2^16 = 0.0000152587890625
    let a3_quantized = Quantized { x: a3 };
    let b3_quantized = Quantized { x: b3 };
    // mul = 17592186044415.9999847412109375 * 0.0000152587890625 = 268435455.99999999976716935634613037109375
    let mult3_quantized = a3_quantized * b3_quantized;
    // mul * 2^16 = 17592186044415.9999847412109375
    assert(mult3_quantized.x == 17592186044415);

    // Test case 4: Positive and negative value
    let a4 = 40000; // Original value 40000/2^16 = 0.6103515625
    let b4 = -30000; // Original value -0.457763671875
    let a4_quantized = Quantized { x: a4 };
    let b4_quantized = Quantized { x: b4 };

    // 0.6103515625 * -0.457763671875 = -0.2793967723846435546875
    let mult4_quantized = a4_quantized * b4_quantized;
    // -0.2793967723846435546875 * 2^16 = -18310.546875
    assert(mult4_quantized.x == -18310);

    // Test case 5: Both values negative
    let a5 = -40000; // Original value -40000 / 2^16 = -0.6103515625
    let b5 = -20000; // -0.30517578125
    let a5_quantized = Quantized { x: a5 };
    let b5_quantized = Quantized { x: b5 };
    // -0.6103515625 *  -0.30517578125 = 0.186264514923095703125
    let mult5_quantized = a5_quantized * b5_quantized;
    // Expected result: 12207.03125
    assert(mult5_quantized.x == 12207);
}

#[test]
fn test_sub() {
    let a = 1152921504606846975;
    let b = 1152921504606846975;
    let a_quantized = Quantized { x: a };
    let b_quantized = Quantized { x: b };
    let addition_quantized = a_quantized - b_quantized;
    assert(addition_quantized.x == 0);

    let a1 = 1152921504606846975;
    let b1 = 0;
    let a1_quantized = Quantized { x: a1 };
    let b1_quantized = Quantized { x: b1 };
    let addition1_quantized = a1_quantized - b1_quantized;
    assert(addition1_quantized.x == a1);

    let a1 = 1152921504606846975;
    let b1 = 0;
    let a1_quantized = Quantized { x: a1 };
    let b1_quantized = Quantized { x: b1 };
    let addition1_quantized = b1_quantized - a1_quantized;
    // The result should be -a1, which means it is p - a1 in the field because
    // of the wrap around.
    let result = 21888242871839275222246405745257275088548364400416034343697051265071201648642;
    assert(addition1_quantized.x == result);

    let a2 = 12345;
    let b2 = 67890;
    let a2_quantized = Quantized { x: a2 };
    let b2_quantized = Quantized { x: b2 };
    let addition2_quantized = a2_quantized + b2_quantized;
    assert(addition2_quantized.x == (a2 + b2));
}
