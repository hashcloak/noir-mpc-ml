use std::ops::{Add, Div, Mul, Sub};
use std::cmp::Ordering;

pub struct Quantized {
    pub x: U128,
    pub neg: bool
}

impl Quantized {

    pub fn zero() -> Self {
        Quantized { x: U128::zero(), neg: false }
    }

    // only works for x max 64 bits
    pub fn new_pos(x: Field) -> Self {
      x.assert_max_bit_size::<64>();
      Quantized { x: U128::from_u64s_le(x as u64,0), neg: false }
    }

    // only works for x max 64 bits
    pub fn new_neg(x: Field) -> Self {
      x.assert_max_bit_size::<64>();
      Quantized { x: U128::from_u64s_le(x as u64,0), neg: true }
    }


    fn add(self: Self, other: Self) -> Self {
      // Q: if in any of the branches there would be an overflow, does assert_eq always fail?
        let mut res = Quantized::zero();
        if self.neg & other.neg {
            res = Quantized { x: self.x + other.x, neg: true }; // U128 additions assert_eqs whether this overflows
        } else if self.neg { // -a + b
            if other.x >= self.x {
              res = Quantized { x: other.x - self.x, neg: false };
            } else {
              res = Quantized { x: self.x - other.x, neg: true };
            }
        } else if other.neg { // a-b
            if self.x >= other.x {
              res = Quantized { x: self.x - other.x, neg: false };
            } else {
              res = Quantized { x: other.x - self.x, neg: true };
            }
        } else {
          res = Quantized { x: other.x + self.x, neg: false }; // not doing this at the beginning in case there is overflow
        }
        res
    }

    fn sub(self: Self, other: Self) -> Self {
        let neg_other = Quantized { x: other.x, neg: !other.neg };
        self.add(neg_other)
    }

    fn mul(self: Self, other: Self) -> Self {
        let res_x = (self.x * other.x) / U128::from_u64s_le(65536 as u64, 0);  // U128 multiplication, will overflow if needed. Divide by 2^16 to adjust scaling
        let res_neg = self.neg ^ other.neg; // XOR to determine sign

        Quantized { x: res_x, neg: res_neg }
    }
}


impl Mul for Quantized {
    fn mul(self, other: Self) -> Self {
        self.mul(other)
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

impl Ord for Quantized {
    fn cmp(self: Self, other: Self) -> Ordering {
        if self.neg != other.neg {
            if self.neg { Ordering::less() } else { Ordering::greater() }
        }

        if self.neg {
            other.x.cmp(self.x) // Negative values: larger magnitude means smaller value
        } else {
            self.x.cmp(other.x) // Positive values: normal comparison
        }
    }
}

// tests pass but when running gatecount U128 gives issues with private fields hi & lo

// #[test]
// fn test_add() {
//     // Test case 1: Max value 2^60-1, adding positive and negative to get zero
//     let a = Quantized { x: U128 { lo: 1152921504606846975, hi: 0 }, neg: false }; // 2^60 - 1
//     let b = Quantized { x: U128 { lo: 1152921504606846975, hi: 0 }, neg: true };  // - (2^60 - 1)
//     let result = a + b;
//     assert_eq(result.x, U128 { lo: 0, hi: 0 });
//     assert_eq(result.neg, false);

//     // Test case 2: Adding a positive number and zero
//     let a = Quantized { x: U128 { lo: 1152921504606846975, hi: 0 }, neg: false };
//     let b = Quantized { x: U128 { lo: 0, hi: 0 }, neg: false };
//     let result = a + b;
//     assert_eq(result.x, a.x);
//     assert_eq(result.neg, false);

//     // Test case 3: Adding small positive values
//     let a = Quantized { x: U128 { lo: 12345, hi: 0 }, neg: false };
//     let b = Quantized { x: U128 { lo: 67890, hi: 0 }, neg: false };
//     let result = a + b;
//     assert_eq(result.x, U128 { lo: 12345 + 67890, hi: 0 });
//     assert_eq(result.neg, false);

//     // Test case 4: Positive and negative values where positive is larger
//     let a = Quantized { x: U128 { lo: 50000, hi: 0 }, neg: false };
//     let b = Quantized { x: U128 { lo: 30000, hi: 0 }, neg: true }; // -30000
//     let result = a + b;
//     assert_eq(result.x, U128 { lo: 20000, hi: 0 });
//     assert_eq(result.neg, false);

//     // Test case 5: Positive and negative values where negative is larger
//     let a = Quantized { x: U128 { lo: 30000, hi: 0 }, neg: false };
//     let b = Quantized { x: U128 { lo: 50000, hi: 0 }, neg: true }; // -50000
//     let result = a + b;
//     assert_eq(result.x, U128 { lo: 20000, hi: 0 });
//     assert_eq(result.neg, true);

//     // Test case 6: Adding two negative values
//     let a = Quantized { x: U128 { lo: 40000, hi: 0 }, neg: true }; // -40000
//     let b = Quantized { x: U128 { lo: 20000, hi: 0 }, neg: true }; // -20000
//     let result = a + b;
//     assert_eq(result.x, U128 { lo: 60000, hi: 0 });
//     assert_eq(result.neg, true);
// }

// #[test(should_fail)]
// fn test_overflow_addition1() {
//     // This test fails because the sum exceeds the U128 limit
//     let a = Quantized { x: U128 { lo: 0xFFFFFFFFFFFFFFFF, hi: 0xFFFFFFFFFFFFFFFF }, neg: false }; // Max U128 value
//     let b = Quantized { x: U128 { lo: 1, hi: 0 }, neg: false }; // Smallest possible increment
//     let result = a + b; // Should cause overflow
// }

// #[test(should_fail)]
// fn test_overflow_addition2() {
//     // Adding two large numbers that together exceed U128 capacity
//     let a = Quantized { x: U128 { lo: 0x8000000000000000, hi: 0x8000000000000000 }, neg: false }; // Half of max U128
//     let b = Quantized { x: U128 { lo: 0x8000000000000000, hi: 0x8000000000000000 }, neg: false }; // Another half
//     let result = a + b; // Should overflow
// }

// #[test]
// fn test_sub() {
//     // Test case 1: Subtracting a number from itself should yield zero
//     let a = Quantized { x: U128 { lo: 1152921504606846975, hi: 0 }, neg: false };
//     let b = Quantized { x: U128 { lo: 1152921504606846975, hi: 0 }, neg: false };
//     let result = a - b;
//     assert_eq(result.x, U128 { lo: 0, hi: 0 });
//     assert_eq(result.neg, false);

//     // Test case 2: Subtracting zero should not change the value
//     let a1 = Quantized { x: U128 { lo: 1152921504606846975, hi: 0 }, neg: false };
//     let b1 = Quantized { x: U128 { lo: 0, hi: 0 }, neg: false };
//     let result1 = a1 - b1;
//     assert_eq(result1.x, a1.x);
//     assert_eq(result1.neg, false);

//     // Test case 3: Subtracting a number from zero should yield a negative result
//     let a1 = Quantized { x: U128 { lo: 1152921504606846975, hi: 0 }, neg: false };
//     let b1 = Quantized { x: U128 { lo: 0, hi: 0 }, neg: false };
//     let result1 = b1 - a1;
//     assert_eq(result1.x, U128 { lo: 1152921504606846975, hi: 0 });
//     assert_eq(result1.neg, true);

//     // Test case 4: Subtracting a larger number from a smaller one
//     let a2 = Quantized { x: U128 { lo: 12345, hi: 0 }, neg: false };
//     let b2 = Quantized { x: U128 { lo: 67890, hi: 0 }, neg: false };
//     let result2 = a2 - b2;
//     assert_eq(result2.x, U128 { lo: 55545, hi: 0 });
//     assert_eq(result2.neg, true);

//     // Test case 5: Subtracting a negative number (equivalent to addition)
//     let a3 = Quantized { x: U128 { lo: 50000, hi: 0 }, neg: false };
//     let b3 = Quantized { x: U128 { lo: 30000, hi: 0 }, neg: true }; // -30000
//     let result3 = a3 - b3;
//     assert_eq(result3.x, U128 { lo: 80000, hi: 0 });
//     assert_eq(result3.neg, false);

//     // Test case 6: Subtracting a positive number from a negative one
//     let a4 = Quantized { x: U128 { lo: 30000, hi: 0 }, neg: true };  // -30000
//     let b4 = Quantized { x: U128 { lo: 50000, hi: 0 }, neg: false }; // +50000
//     let result4 = a4 - b4;
//     assert_eq(result4.x, U128 { lo: 80000, hi: 0 });//-8000
//     assert_eq(result4.neg, true);

//     // Test case 7: Subtracting a large negative from a smaller negative (should become positive)
//     let a5 = Quantized { x: U128 { lo: 30000, hi: 0 }, neg: true };  // -30000
//     let b5 = Quantized { x: U128 { lo: 70000, hi: 0 }, neg: true };  // -70000
//     let result5 = a5 - b5;
//     assert_eq(result5.x, U128 { lo: 40000, hi: 0 });
//     assert_eq(result5.neg, false);

//     // Test case 8: 0-1=-1
//     let a = Quantized { x: U128 { lo: 0, hi: 0 }, neg: false };
//     let b = Quantized { x: U128 { lo: 1, hi: 0 }, neg: false };
//     let result = a - b;
    
//     assert_eq(result.x, U128 { lo: 1, hi: 0 });
//     assert_eq(result.neg, true);
// }

// #[test]
// fn test_mul() {
//     // Test case 1: Small positive values
//     let a1 = Quantized { x: U128 { lo: 12345, hi: 0 }, neg: false };
//     let b1 = Quantized { x: U128 { lo: 67890, hi: 0 }, neg: false };
//     let result1 = a1 * b1;
//     assert_eq(result1.x, U128 { lo: 12788, hi: 0 });
//     assert_eq(result1.neg, false);

//     // Test case 2: Large positive values
//     let a2 = Quantized { x: U128 { lo: 1152921504606846975, hi: 0 }, neg: false }; // 2^60 - 1
//     let b2 = Quantized { x: U128 { lo: 2, hi: 0 }, neg: false };
//     let result2 = a2 * b2;
//     assert_eq(result2.x, U128 { lo: 35184372088831, hi: 0 });
//     assert_eq(result2.neg, false);

//     // Test case 3: Large positive and small positive value
//     let a3 = Quantized { x: U128 { lo: 1152921504606846975, hi: 0 }, neg: false }; // 2^60 - 1
//     let b3 = Quantized { x: U128 { lo: 1, hi: 0 }, neg: false };
//     let result3 = a3 * b3;
//     assert_eq(result3.x, U128 { lo: 17592186044415, hi: 0 });
//     assert_eq(result3.neg, false);

//     // Test case 4: Positive and negative value
//     let a4 = Quantized { x: U128 { lo: 40000, hi: 0 }, neg: false };
//     let b4 = Quantized { x: U128 { lo: 30000, hi: 0 }, neg: true }; // Negative
//     let result4 = a4 * b4;
//     assert_eq(result4.x, U128 { lo: 18310, hi: 0 });
//     assert_eq(result4.neg, true);

//     // Test case 5: Both values negative
//     let a5 = Quantized { x: U128 { lo: 40000, hi: 0 }, neg: true }; // Negative
//     let b5 = Quantized { x: U128 { lo: 20000, hi: 0 }, neg: true }; // Negative
//     let result5 = a5 * b5;
//     assert_eq(result5.x, U128 { lo: 12207, hi: 0 });
//     assert_eq(result5.neg, false);
// }

// #[test]
// fn test_order() {
//     // Test 1: comparison between positive and negative value.
//     // a = 0.2 and b = -0.2
//     let a_quantized = Quantized::new_pos(13107);
//     let b_quantized = Quantized::new_neg(13107);
//     assert(a_quantized >= b_quantized);

//     // Test 2: comparison between two positive numbers.
//     // a = 1 and b = 0.2.
//     let a_quantized = Quantized::new_pos(65536);
//     let b_quantized = Quantized::new_pos(13107);
//     assert(a_quantized > b_quantized);

//     // Test 3: comparison between two positive numbers where the roles are
//     // inverted.
//     let a_quantized = Quantized::new_pos(65536);
//     let b_quantized = Quantized::new_pos(78643);
//     assert(a_quantized < b_quantized);

//     // Test 4: comparison between two negative numbers.
//     let a_quantized = Quantized::new_neg(65536);
//     let b_quantized = Quantized::new_neg(13107);
//     assert(a_quantized < b_quantized);
// }
