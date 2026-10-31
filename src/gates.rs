//! Boolean bitwise arithmetic with explicit overflow detection.
use std::io::{self, ErrorKind};
pub fn add(mut a: u128, mut b: u128) -> io::Result<u128> {
    while b != 0 {
        if (a & b) >> 127 != 0 {
            return Err(io::Error::new(ErrorKind::InvalidData, "addition overflow"));
        }
        let carry = (a & b) << 1;
        a ^= b;
        b = carry;
    }
    Ok(a)
}
pub fn ge(a: u128, b: u128) -> bool {
    let mut eq = true;
    let mut greater = false;
    for shift in (0..128).rev() {
        let left = (a >> shift) & 1 != 0;
        let right = (b >> shift) & 1 != 0;
        greater |= eq & left & !right;
        eq &= !(left ^ right);
    }
    greater | eq
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gate_equivalence() {
        for a in 0..256_u128 {
            for b in 0..256_u128 {
                assert_eq!(add(a, b).unwrap(), a + b);
                assert_eq!(ge(a, b), a >= b);
            }
        }
    }
    #[test]
    fn boundaries() {
        assert!(add(u128::MAX, 1).is_err());
        assert_eq!(add(u128::MAX - 1, 1).unwrap(), u128::MAX);
        assert!(ge(u128::MAX, 0));
        assert!(!ge(0, u128::MAX));
    }
}
