//! Physical old/full versus new/exact-cached inference at 6,250,000 bits.
//!
//! The pre-existing public FiveCircleHierarchy is bounded at 625 inputs.
//! Both arms therefore run in the SAME explicitly generalized five-ary
//! representation, with identical compiled 32-entry Walsh truth tables:
//! OLD = re-evaluate every node; NEW = cache unchanged subtrees and
//! recompute only changed dependency cones. All 64 lanes are bitwise checked.
//! Exactly 6,250,000 real input positions; five upper-level padding
//! slots are hard-coded FALSE, not counted as inputs or physical contacts.
use std::time::Instant;

use v888_booleen::fourier5_hierarchy::GateKind;
use v888_booleen::fourier5_wide::{
    WideEdit, WideFiveCache, WideFiveTree, SIX_MILLION_INPUTS,
};

struct Xorshift(u64);
impl Xorshift {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0x9e37_79b9_7f4a_7c15 } else { seed })
    }

    fn next(&mut self) -> u64 {
        let mut value = self.0;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.0 = value;
        value
    }
}

fn linux_hwm_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status
        .lines()
        .find(|line| line.starts_with("VmHWM:"))
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|value| value.parse::<u64>().ok())
}

fn flipped_parity() -> v888_booleen::fourier5::FiveCircle {
    let original = GateKind::Parity.five_circle();
    let mut circles = *original.circles();
    circles[0].phase_pi = !circles[0].phase_pi;
    v888_booleen::fourier5::FiveCircle::new(circles, original.bias())
        .expect("existing 5-mask Fourier model")
}

fn profile(width: usize, seed: u64) -> Result<(), String> {
    let tree = WideFiveTree::new(width).map_err(|err| format!("{err:?}"))?;
    let mut random = Xorshift::new(seed);
    let started = Instant::now();
    let planes: Vec<u64> = (0..width).map(|_| random.next()).collect();
    let generate_us = started.elapsed().as_micros();

    let started = Instant::now();
    let old_baseline = tree.evaluate_full(&planes, None)
        .map_err(|err| format!("{err:?}"))?;
    let old_baseline_us = started.elapsed().as_micros();
    let started = Instant::now();
    let cache = WideFiveCache::new(&tree, &planes)
        .map_err(|err| format!("{err:?}"))?;
    let cache_build_us = started.elapsed().as_micros();
    let started = Instant::now();
    let independent = tree.independent_reference(&planes)
        .map_err(|err| format!("{err:?}"))?;
    let independent_oracle_us = started.elapsed().as_micros();
    if old_baseline.output != cache.output() || old_baseline.output != independent {
        return Err("baseline and fully independent Boolean oracle disagree".into());
    }

    let middle = tree.depth() / 2;
    let last = tree.depth() - 1;
    let edits = [
        ("bottom_walsh_gain_bias", WideEdit::Parameter {
            layer: 0, neuron: 0,
            model: GateKind::And.five_circle(),
        }),
        ("distant_source_swap", WideEdit::SourceSwap {
            first: 0, second: width / 2,
        }),
        ("middle_phase_flip", WideEdit::Parameter {
            layer: middle,
            neuron: tree.level_widths()[middle] / 2,
            model: flipped_parity(),
        }),
        ("root_phase_flip", WideEdit::Parameter {
            layer: last, neuron: 0,
            model: flipped_parity(),
        }),
    ];

    let mut total_old_us = old_baseline_us;
    let mut total_cached_us = cache_build_us;
    let mut old_gates = old_baseline.gate_evaluations;
    let mut new_gates = cache.baseline_gate_evaluations();
    for (number, &(name, edit)) in edits.iter().enumerate() {
        let full_first = (number + (seed as usize & 1)) % 2 == 0;
        let (full, full_us, delta, delta_us) = if full_first {
            let start = Instant::now();
            let previous = tree.evaluate_full(&planes, Some(edit))
                .map_err(|err| format!("{err:?}"))?;
            let old_us = start.elapsed().as_micros();
            let start = Instant::now();
            let current = cache.evaluate_delta(edit)
                .map_err(|err| format!("{err:?}"))?;
            let new_us = start.elapsed().as_micros();
            (previous, old_us, current, new_us)
        } else {
            let start = Instant::now();
            let current = cache.evaluate_delta(edit)
                .map_err(|err| format!("{err:?}"))?;
            let new_us = start.elapsed().as_micros();
            let start = Instant::now();
            let previous = tree.evaluate_full(&planes, Some(edit))
                .map_err(|err| format!("{err:?}"))?;
            let old_us = start.elapsed().as_micros();
            (previous, old_us, current, new_us)
        };
        if full.output != delta.output {
            return Err(format!(
                "not bitwise exact across 64 lanes for {name}, width={width}, seed={seed}"
            ));
        }
        if full.gate_evaluations != tree.logical_neurons() {
            return Err("full recomputation did not evaluate all nodes".into());
        }
        if delta.gate_evaluations > 2 * tree.depth() {
            return Err("dependency cone exceeded two paths".into());
        }
        total_old_us += full_us;
        total_cached_us += delta_us;
        old_gates += full.gate_evaluations;
        new_gates += delta.gate_evaluations;
        println!(
            "WIDE_EDIT input_bits={width} seed={seed} name={name} full_us={full_us} cached_us={delta_us} full_gates={} cached_delta_gates={} all_64_lanes_equal=true changed_lane_count={}",
            full.gate_evaluations, delta.gate_evaluations,
            (full.output ^ old_baseline.output).count_ones()
        );
    }

    let gate_savings = 100.0 * (1.0 - new_gates as f64 / old_gates as f64);
    let speedup = total_old_us as f64 / total_cached_us as f64;
    println!(
        "WIDE_SUMMARY input_bits={width} seed={seed} lanes=64 depth={} neurons={} contacts={} padding_slots={} source_bytes={} cached_node_bytes={} full_total_us={total_old_us} cached_total_us={total_cached_us} full_baseline_us={old_baseline_us} cache_build_us={cache_build_us} full_gates={old_gates} cached_gates={new_gates} gate_savings_percent={gate_savings:.4} speedup={speedup:.4} sample_generation_us={generate_us} independent_oracle_us={independent_oracle_us} peak_rss_kib={} exact=true",
        tree.depth(), tree.logical_neurons(), tree.real_contacts(),
        tree.padding_slots(), width * 8, tree.cache_word_bytes(),
        linux_hwm_kib().map_or_else(|| "unavailable".to_owned(), |v| v.to_string())
    );
    Ok(())
}

fn main() -> Result<(), String> {
    // Three fixed independent random seeds, matched baseline and intervention
    // stream at the small 625-bit control and requested 6,250,000-bit scale.
    for seed in [0x6250_0001_u64, 0x6250_0002, 0x6250_0003] {
        profile(625, seed)?;
        profile(SIX_MILLION_INPUTS, seed)?;
    }
    println!("FOURIER5_WIDE_6250000_FULL_VS_CACHED_EXACT_PASS");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaled_reference_matches_independent_boolean_oracle_and_all_edits() {
        profile(625, 0x55_1212).unwrap();
    }
}
