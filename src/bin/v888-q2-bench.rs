//! Synthetic, bounded GF(4) bitplane benchmark and independent scalar oracle.
//!
//! Results are timing probes, not energy measurements or biological evidence.
use std::hint::black_box;
use std::time::Instant;
use v888_booleen::packed_q2::PackedQ2;

fn gf4_scalar(a: u8, b: u8) -> u8 {
    let (a0, a1, b0, b1) = (a & 1, a >> 1, b & 1, b >> 1);
    ((a0 & b0) ^ (a1 & b1)) | (((a0 & b1) ^ (a1 & b0) ^ (a1 & b1)) << 1)
}

fn scalar_dot(left: &[u8], right: &[u8]) -> u8 {
    left.iter()
        .zip(right)
        .fold(0_u8, |acc, (&a, &b)| acc ^ gf4_scalar(a, b))
}

fn scalar_mul_add_assign(output: &mut [u8], left: &[u8], right: &[u8]) {
    for ((out, &a), &b) in output.iter_mut().zip(left).zip(right) {
        *out ^= gf4_scalar(a, b);
    }
}

fn run_case(len: usize, repeats: usize) {
    assert!(repeats > 0);
    let a: Vec<u8> = (0..len).map(|i| ((i * 17 + i / 5) % 4) as u8).collect();
    let b: Vec<u8> = (0..len).map(|i| ((i * 13 + i / 7 + 1) % 4) as u8).collect();
    let initial: Vec<u8> = (0..len)
        .map(|i| ((i * 19 + i / 11 + 2) % 4) as u8)
        .collect();

    let packing_start = Instant::now();
    let left = PackedQ2::from_levels(&a).expect("valid Q2 source");
    let right = PackedQ2::from_levels(&b).expect("valid Q2 source");
    let base = PackedQ2::from_levels(&initial).expect("valid Q2 accumulator");
    let packing_ns = packing_start.elapsed().as_nanos();

    let expected_dot = scalar_dot(&a, &b);
    assert_eq!(left.gf4_dot(&right), Ok(expected_dot));
    let mut fused_once = base.clone();
    fused_once
        .gf4_mul_add_assign(&left, &right)
        .expect("equal shapes");
    let mut scalar_once = initial.clone();
    scalar_mul_add_assign(&mut scalar_once, &a, &b);
    for (i, &expected) in scalar_once.iter().enumerate() {
        assert_eq!(fused_once.get(i), Ok(expected));
    }

    let mut scalar_checksum = 0_u8;
    let start = Instant::now();
    for _ in 0..repeats {
        scalar_checksum ^= black_box(scalar_dot(black_box(&a), black_box(&b)));
    }
    let scalar_dot_ns = start.elapsed().as_nanos() / repeats as u128;

    let mut packed_checksum = 0_u8;
    let start = Instant::now();
    for _ in 0..repeats {
        packed_checksum ^= black_box(
            left.gf4_dot(black_box(&right))
                .expect("same-length Q2 vectors"),
        );
    }
    let packed_dot_ns = start.elapsed().as_nanos() / repeats as u128;
    assert_eq!(scalar_checksum, packed_checksum);

    let mut scalar_acc = initial.clone();
    let start = Instant::now();
    for _ in 0..repeats {
        scalar_mul_add_assign(black_box(&mut scalar_acc), &a, &b);
    }
    let scalar_mac_ns = start.elapsed().as_nanos() / repeats as u128;

    let mut packed_acc = base;
    let start = Instant::now();
    for _ in 0..repeats {
        packed_acc
            .gf4_mul_add_assign(black_box(&left), black_box(&right))
            .expect("equal shapes");
    }
    let packed_mac_ns = start.elapsed().as_nanos() / repeats as u128;
    for (i, &expected) in scalar_acc.iter().enumerate() {
        assert_eq!(packed_acc.get(i), Ok(expected));
    }

    println!(
        "V888_Q2_BENCH lanes={len} repeats={repeats} packing_three_vectors_ns={packing_ns} \
scalar_dot_ns_per_call={scalar_dot_ns} packed_dot_ns_per_call={packed_dot_ns} \
scalar_mac_ns_per_call={scalar_mac_ns} packed_mac_ns_per_call={packed_mac_ns} \
payload_bytes_three_vectors={}",
        packed_acc.allocated_payload_bytes() * 3
    );
}

fn main() {
    for (len, repeats) in [
        (0, 1024),
        (63, 1024),
        (64, 1024),
        (65, 1024),
        (4096, 256),
        (188_508, 64),
    ] {
        run_case(len, repeats);
    }
}

#[cfg(test)]
mod tests {
    use super::{gf4_scalar, scalar_dot, scalar_mul_add_assign};

    #[test]
    fn all_single_lane_field_pairs_agree_with_scalar_dot_and_accumulation() {
        for a in 0..4 {
            for b in 0..4 {
                assert_eq!(scalar_dot(&[a], &[b]), gf4_scalar(a, b));
                let mut out = [3_u8];
                scalar_mul_add_assign(&mut out, &[a], &[b]);
                assert_eq!(out[0], 3 ^ gf4_scalar(a, b));
            }
        }
    }
}
