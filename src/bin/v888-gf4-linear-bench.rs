//! Paired, bounded CPU experiment for precomposed packed GF(4) linear maps.
//!
//! This measures synthetic CPU latency only; no RAPL, BANC, or model claims.
use std::hint::black_box;
use std::time::Instant;
use v888_booleen::gf4_linear::PackedGf4Linear;
use v888_booleen::packed_q2::PackedQ2;

fn matrix(rows: usize, cols: usize, seed: usize) -> PackedGf4Linear {
    let coefficients = (0..rows)
        .map(|row| {
            let values = (0..cols)
                .map(|col| ((row * 17 + col * 13 + col / 5 + seed) % 4) as u8)
                .collect::<Vec<_>>();
            PackedQ2::from_levels(&values).expect("valid GF4 coefficients")
        })
        .collect::<Vec<_>>();
    PackedGf4Linear::from_rows(cols, coefficients).expect("rectangular GF4 matrix")
}

fn median(values: &mut [u128]) -> u128 {
    values.sort_unstable();
    values[values.len() / 2]
}

fn two_pass(
    first: &PackedGf4Linear,
    second: &PackedGf4Linear,
    input: &PackedQ2,
    repetitions: usize,
) -> u128 {
    let start = Instant::now();
    for _ in 0..repetitions {
        let intermediate = black_box(first.apply(black_box(input)).expect("valid first layer"));
        let output = black_box(
            second
                .apply(black_box(&intermediate))
                .expect("valid second layer"),
        );
        black_box(output);
    }
    start.elapsed().as_nanos() / repetitions as u128
}

fn precomposed(fused: &PackedGf4Linear, input: &PackedQ2, repetitions: usize) -> u128 {
    let start = Instant::now();
    for _ in 0..repetitions {
        let output = black_box(fused.apply(black_box(input)).expect("valid fused layer"));
        black_box(output);
    }
    start.elapsed().as_nanos() / repetitions as u128
}

fn run_case(inputs: usize, hidden: usize, outputs: usize, repetitions: usize) {
    let first = matrix(hidden, inputs, 1);
    let second = matrix(outputs, hidden, 2);
    let input_values = (0..inputs)
        .map(|i| ((i * 19 + i / 11 + 3) % 4) as u8)
        .collect::<Vec<_>>();
    let input = PackedQ2::from_levels(&input_values).expect("valid input");

    let build_start = Instant::now();
    let fused = second.compose(&first).expect("matching matrix dimensions");
    let compose_ns = build_start.elapsed().as_nanos();
    let intermediate = first.apply(&input).expect("first layer");
    let expected = second.apply(&intermediate).expect("second layer");
    assert_eq!(fused.apply(&input), Ok(expected));

    // Alternate measurement order to reduce a fixed first-path advantage.
    let mut baseline_samples = Vec::with_capacity(7);
    let mut composed_samples = Vec::with_capacity(7);
    for trial in 0..7 {
        if trial % 2 == 0 {
            baseline_samples.push(two_pass(&first, &second, &input, repetitions));
            composed_samples.push(precomposed(&fused, &input, repetitions));
        } else {
            composed_samples.push(precomposed(&fused, &input, repetitions));
            baseline_samples.push(two_pass(&first, &second, &input, repetitions));
        }
    }

    let baseline_ns = median(&mut baseline_samples);
    let composed_ns = median(&mut composed_samples);
    assert_eq!(fused.apply(&input), Ok(expected));

    println!(
        concat!(
            "V888_GF4_LINEAR inputs={} hidden={} outputs={} ",
            "repetitions={} trials=7 compose_once_ns={} ",
            "two_pass_median_ns_per_call={} precomposed_median_ns_per_call={} ",
            "separate_coeff_payload_bytes={} composed_coeff_payload_bytes={} ",
            "intermediate_payload_bytes={}"
        ),
        inputs,
        hidden,
        outputs,
        repetitions,
        compose_ns,
        baseline_ns,
        composed_ns,
        first.coefficient_payload_bytes() + second.coefficient_payload_bytes(),
        fused.coefficient_payload_bytes(),
        intermediate.allocated_payload_bytes(),
    );
}

fn main() {
    // Include non-aligned bitplane boundaries and a denser case.
    for (inputs, hidden, outputs, repetitions) in [
        (63, 64, 65, 128),
        (65, 65, 63, 128),
        (129, 64, 65, 128),
        (256, 128, 64, 64),
        // A low-rank bottleneck can make the composed matrix much larger.
        (1024, 8, 128, 32),
    ] {
        run_case(inputs, hidden, outputs, repetitions);
    }
}

#[cfg(test)]
mod tests {
    use super::{matrix, median};

    #[test]
    fn median_and_composed_identity_shapes() {
        let mut samples = [9, 3, 5, 1, 7];
        assert_eq!(median(&mut samples), 5);

        let first = matrix(65, 63, 1);
        let second = matrix(129, 65, 2);
        let combined = second.compose(&first).unwrap();
        assert_eq!(combined.input_len(), 63);
        assert_eq!(combined.output_len(), 129);
        assert_eq!(combined.coefficient_payload_bytes(), 129 * 16);
    }
}
