//! Paired synthetic CPU timings for the exact packed GF(4) inner product.
//! This is a classical Rust kernel benchmark, not an energy measurement or
//! evidence of faster Boolean graph propagation. All data stay in memory.
use std::env;
use std::hint::black_box;
use std::time::Instant;

use v888_booleen::packed_q2::PackedQ2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Config {
    lanes: usize,
    repetitions: usize,
}

impl Config {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let mut config = Self {
            lanes: 188_508,
            repetitions: 11,
        };
        for arg in args {
            let (name, value) = arg
                .strip_prefix("--")
                .and_then(|s| s.split_once('='))
                .ok_or_else(|| format!("invalid option: {arg}"))?;
            match name {
                "lanes" => {
                    config.lanes = value
                        .parse()
                        .map_err(|_| format!("invalid lane count: {value}"))?;
                }
                "repetitions" => {
                    config.repetitions = value
                        .parse()
                        .map_err(|_| format!("invalid repetitions: {value}"))?;
                }
                _ => return Err(format!("unrecognized option: {name}")),
            }
        }
        if !(1..=1_048_576).contains(&config.lanes) {
            return Err("lanes must be within 1..=1048576".to_owned());
        }
        if !(3..=64).contains(&config.repetitions) {
            return Err("repetitions must be within 3..=64".to_owned());
        }
        Ok(config)
    }
}

/// GF(4) product in the polynomial basis x²+x+1 for codes 0,1,x,1+x.
fn scalar_mul(a: u8, b: u8) -> u8 {
    let low = (a & 1) & (b & 1) ^ ((a >> 1) & (b >> 1));
    let high = ((a & 1) & (b >> 1))
        ^ ((a >> 1) & (b & 1))
        ^ ((a >> 1) & (b >> 1));
    low | (high << 1)
}

fn scalar_dot(a: &[u8], b: &[u8]) -> u8 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(0_u8, |total, (&x, &y)| total ^ scalar_mul(x, y))
}

/// Synthetic and deterministic four-level data; no anatomical interpretation.
fn fixture(lanes: usize) -> (Vec<u8>, Vec<u8>) {
    let a = (0..lanes)
        .map(|i| ((i.wrapping_mul(17) ^ i.rotate_left(3) ^ (i >> 5)) & 3) as u8)
        .collect();
    let b = (0..lanes)
        .map(|i| ((i.wrapping_mul(13) ^ i.rotate_left(1) ^ (i >> 3) ^ 3) & 3) as u8)
        .collect();
    (a, b)
}

fn measure<T>(work: impl FnOnce() -> T) -> (T, u128) {
    let started = Instant::now();
    let value = black_box(work());
    (value, started.elapsed().as_nanos())
}

#[derive(Debug, PartialEq, Eq)]
struct Evidence {
    lanes: usize,
    scalar_field_code: u8,
    packed_field_code: u8,
    scalar_payload_bytes: usize,
    packed_payload_bytes: usize,
    scalar_median_ns: u128,
    packed_median_ns: u128,
}

fn benchmark(config: Config) -> Evidence {
    let (a, b) = fixture(config.lanes);
    // Packing is deliberately excluded from the inner-product timing, but
    // both input memory payload sizes are reported. This measures a reused
    // packed representation, NOT an end-to-end encode+compute+decode path.
    let packed_a = PackedQ2::from_levels(&a).expect("four-level fixture");
    let packed_b = PackedQ2::from_levels(&b).expect("four-level fixture");
    let scalar = scalar_dot(&a, &b);
    let packed = packed_a.gf4_dot(&packed_b).expect("matching packed lengths");
    assert_eq!(scalar, packed, "independent GF(4) reference mismatch");

    // Warmups and reference comparison are outside the timed trials.
    black_box(scalar_dot(black_box(&a), black_box(&b)));
    black_box(packed_a.gf4_dot(black_box(&packed_b)).expect("warmup parity"));
    let mut scalar_ns = Vec::with_capacity(config.repetitions);
    let mut packed_ns = Vec::with_capacity(config.repetitions);
    for rep in 0..config.repetitions {
        // Pair on the same inputs and alternate execution order.
        let (s, p) = if rep % 2 == 0 {
            let s = measure(|| scalar_dot(black_box(&a), black_box(&b)));
            let p = measure(|| packed_a.gf4_dot(black_box(&packed_b)).unwrap());
            (s, p)
        } else {
            let p = measure(|| packed_a.gf4_dot(black_box(&packed_b)).unwrap());
            let s = measure(|| scalar_dot(black_box(&a), black_box(&b)));
            (s, p)
        };
        assert_eq!(s.0, scalar);
        assert_eq!(p.0, packed);
        scalar_ns.push(s.1);
        packed_ns.push(p.1);
    }
    scalar_ns.sort_unstable();
    packed_ns.sort_unstable();
    let idx = (config.repetitions - 1) / 2;
    Evidence {
        lanes: config.lanes,
        scalar_field_code: scalar,
        packed_field_code: packed,
        scalar_payload_bytes: a.len() + b.len(),
        packed_payload_bytes: packed_a.allocated_payload_bytes()
            + packed_b.allocated_payload_bytes(),
        scalar_median_ns: scalar_ns[idx],
        packed_median_ns: packed_ns[idx],
    }
}

fn main() {
    let config = match Config::parse(env::args().skip(1)) {
        Ok(config) => config,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };
    let report = benchmark(config);
    println!(
        "V888_GF4_BENCH lanes={} repetitions={} scalar_median_ns={} \
         packed_median_ns={} scalar_payload_bytes={} packed_payload_bytes={} \
         scalar_code={} packed_code={} exact_parity=true \
         excludes_packing=true synthetic_data=true energy_measured=false",
        report.lanes,
        config.repetitions,
        report.scalar_median_ns,
        report.packed_median_ns,
        report.scalar_payload_bytes,
        report.packed_payload_bytes,
        report.scalar_field_code,
        report.packed_field_code,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_bounds_are_strict() {
        assert!(Config::parse(["--lanes=0".to_string()]).is_err());
        assert!(Config::parse(["--lanes=1048577".to_string()]).is_err());
        assert!(Config::parse(["--repetitions=2".to_string()]).is_err());
        assert!(Config::parse(["--repetitions=65".to_string()]).is_err());
        assert!(Config::parse(["--unknown=12".to_string()]).is_err());
        assert!(Config::parse(["--lanes=no".to_string()]).is_err());
    }

    #[test]
    fn scalar_field_products_cover_all_sixteen_pairs() {
        for a in 0..4_u8 {
            for b in 0..4_u8 {
                let x = PackedQ2::from_levels(&[a]).unwrap();
                let y = PackedQ2::from_levels(&[b]).unwrap();
                assert_eq!(x.gf4_dot(&y), Ok(scalar_mul(a, b)));
            }
        }
        assert_eq!(scalar_mul(2, 2), 3);
        assert_eq!(scalar_mul(2, 3), 1);
        assert_eq!(scalar_mul(3, 3), 2);
    }

    #[test]
    fn paired_benchmark_is_bit_exact_and_payload_is_bounded() {
        for lanes in [1, 2, 63, 64, 65, 129, 257] {
            let evidence = benchmark(Config {
                lanes,
                repetitions: 3,
            });
            assert_eq!(evidence.scalar_field_code, evidence.packed_field_code);
            assert_eq!(evidence.scalar_payload_bytes, lanes * 2);
            assert_eq!(
                evidence.packed_payload_bytes,
                lanes.div_ceil(64) * 32
            );
        }
    }
}
