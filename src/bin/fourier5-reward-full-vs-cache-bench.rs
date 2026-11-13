//! Paired full-recomputation versus persistent exact-cache TRAIN optimization.
//! Same observed output-only labels, same proposals and accept/reject decisions.
//! Four passages x 64 proposals per seed, three independent synthetic seeds.
//! Includes model initialization and full cache rebuild after every FULL
//! accepted edit; excludes teacher/source generation and heldout scoring.
//! This is NOT a hardware energy or DRAM traffic experiment.
use std::io::{self, Write};
use std::time::Instant;

use v888_booleen::fourier5::FiveCircle;
use v888_booleen::fourier5_hierarchy::GateKind;
use v888_booleen::fourier5_wide::{WideFiveTree, SIX_MILLION_INPUTS};
use v888_booleen::fourier5_wide_learning::WidePlasticCircuit;

const SEEDS: [u64; 3] = [0x9a77_0001, 0x9a77_0002, 0x9a77_0003];
const PASS_COUNT: usize = 4;
const PROPOSALS_PER_PASS: usize = 64;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn words(count: usize, seed: u64) -> Vec<u64> {
    let mut x = seed;
    (0..count)
        .map(|_| {
            x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut y = x;
            y = (y ^ (y >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            y = (y ^ (y >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            y ^ (y >> 31)
        })
        .collect()
}

fn hash(layer: usize, neuron: usize, seed: u64) -> u64 {
    let mut x = (neuron as u64) ^ ((layer as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15)) ^ seed;
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn complement_parity() -> FiveCircle {
    let base = GateKind::Parity.five_circle();
    let mut circles = *base.circles();
    for circle in &mut circles {
        circle.phase_pi = !circle.phase_pi;
    }
    FiveCircle::new(circles, -base.bias()).expect("valid existing Walsh masks")
}

fn choices() -> [FiveCircle; 5] {
    [
        GateKind::Majority.five_circle(),
        GateKind::Parity.five_circle(),
        GateKind::And.five_circle(),
        GateKind::Or.five_circle(),
        complement_parity(),
    ]
}

fn hits(output: u64, label: u64) -> u32 {
    64 - (output ^ label).count_ones()
}

fn valid_hits(circuit: &WidePlasticCircuit<'_>, input: &[u64], label: u64) -> Result<u32, String> {
    let trace = circuit
        .evaluate_trace(input)
        .map_err(|e| format!("{e:?}"))?;
    Ok(hits(trace.last().expect("nonempty trace")[0], label))
}

fn full_attempt(
    circuit: &mut WidePlasticCircuit<'_>,
    layer: usize,
    neuron: usize,
    model: FiveCircle,
    label: u64,
) -> Result<(u128, bool), String> {
    let start = Instant::now();
    let candidate = circuit
        .evaluate_full_edit(layer, neuron, model)
        .map_err(|e| format!("{e:?}"))?;
    let accepted = hits(candidate, label) > hits(circuit.output(), label);
    if accepted {
        // Must rebuild all hidden activations, not apply a cached delta.
        circuit
            .accept_full(layer, neuron, model)
            .map_err(|e| format!("{e:?}"))?;
        if circuit.output() != candidate {
            return Err("full training acceptance changed the predicted output".into());
        }
    }
    Ok((start.elapsed().as_micros(), accepted))
}

fn cached_attempt(
    circuit: &mut WidePlasticCircuit<'_>,
    layer: usize,
    neuron: usize,
    model: FiveCircle,
    label: u64,
) -> Result<(u128, bool, usize, usize), String> {
    let start = Instant::now();
    let proposal = circuit
        .propose(layer, neuron, model)
        .map_err(|e| format!("{e:?}"))?;
    let (gates, reads) = (proposal.gate_evaluations, proposal.child_word_reads);
    let accepted = hits(proposal.output, label) > hits(circuit.output(), label);
    if accepted {
        circuit.accept(proposal).map_err(|e| format!("{e:?}"))?;
    }
    Ok((start.elapsed().as_micros(), accepted, gates, reads))
}

fn experiment(width: usize, seed: u64, passes: usize, proposals: usize) -> Result<(), String> {
    let tree = WideFiveTree::new(width).map_err(|e| format!("{e:?}"))?;
    let train = words(width, seed ^ 0x1111_2222_3333_4444);
    let valid = words(width, seed ^ 0x1357_9bdf_2468_ace0);
    let test = words(width, seed ^ 0x0f0f_f0f0_5a5a_a5a5);
    let options = choices();
    let mut teacher = WidePlasticCircuit::new(&tree, &train).map_err(|e| format!("{e:?}"))?;
    teacher.install_models(|layer, neuron| {
        let bucket = hash(layer, neuron, seed) % 10;
        if layer == 0 {
            options[match bucket {
                0 => 2,
                1 => 3,
                2..=5 => 0,
                _ => 1,
            }]
        } else if bucket % 2 == 0 {
            options[1]
        } else {
            options[4]
        }
    });
    // Only ROOT outputs are given to the learner; VALID/TEST stay held out.
    let target = teacher.output();
    let valid_target = teacher
        .evaluate_trace(&valid)
        .map_err(|e| format!("{e:?}"))?
        .last()
        .unwrap()[0];
    let test_target = teacher
        .evaluate_trace(&test)
        .map_err(|e| format!("{e:?}"))?
        .last()
        .unwrap()[0];
    drop(teacher);
    if [target, valid_target, test_target]
        .iter()
        .any(|x| !(16..=48).contains(&x.count_ones()))
    {
        return Err("unbalanced synthetic target, no performance claim".into());
    }
    let init = Instant::now();
    let mut full = WidePlasticCircuit::new(&tree, &train).map_err(|e| format!("{e:?}"))?;
    let full_init_us = init.elapsed().as_micros();
    let init = Instant::now();
    let mut cached = WidePlasticCircuit::new(&tree, &train).map_err(|e| format!("{e:?}"))?;
    let cached_init_us = init.elapsed().as_micros();
    if full.output() != cached.output() {
        return Err("initial output mismatch".into());
    }
    let mut total_full_us = full_init_us;
    let mut total_cached_us = cached_init_us;
    let mut accepted_total = 0;
    let mut rng = Rng(seed ^ 0xfeed_ba12_8888_0001);
    for pass in 1..=passes {
        let mut pass_full_us = 0;
        let mut pass_cached_us = 0;
        let mut pass_accepted = 0;
        let mut pass_gate_evals = 0;
        let mut pass_child_reads = 0;
        let mut skipped = 0;
        for index in 0..proposals {
            let layer = rng.next() as usize % tree.depth();
            let neuron = rng.next() as usize % tree.level_widths()[layer];
            let model = options[rng.next() as usize % options.len()];
            if model == full.model(layer, neuron).map_err(|e| format!("{e:?}"))? {
                skipped += 1;
                continue;
            }
            // Alternating order mitigates the most obvious warm-cache bias.
            let (f, c) = if (pass + index) % 2 == 0 {
                let f = full_attempt(&mut full, layer, neuron, model, target)?;
                let c = cached_attempt(&mut cached, layer, neuron, model, target)?;
                (f, c)
            } else {
                let c = cached_attempt(&mut cached, layer, neuron, model, target)?;
                let f = full_attempt(&mut full, layer, neuron, model, target)?;
                (f, c)
            };
            if f.1 != c.1 || full.output() != cached.output() {
                return Err("exact paired train decision/output mismatch".into());
            }
            pass_full_us += f.0;
            pass_cached_us += c.0;
            pass_gate_evals += c.2;
            pass_child_reads += c.3;
            pass_accepted += usize::from(f.1);
        }
        total_full_us += pass_full_us;
        total_cached_us += pass_cached_us;
        accepted_total += pass_accepted;
        if full.commits() != cached.commits() || full.commits() != accepted_total {
            return Err("persistent full/delta model versions diverged".into());
        }
        let validation_full = valid_hits(&full, &valid, valid_target)?;
        let validation_cached = valid_hits(&cached, &valid, valid_target)?;
        let testing_full = valid_hits(&full, &test, test_target)?;
        let testing_cached = valid_hits(&cached, &test, test_target)?;
        if validation_full != validation_cached || testing_full != testing_cached {
            return Err("independent heldout full/delta model mismatch".into());
        }
        println!(
            "PAIRED_TRAIN_PASS seed={seed} width={width} pass={pass} attempted={proposals} skipped={skipped} accepted={pass_accepted} accepted_total={accepted_total} train={}/64 valid={validation_full}/64 test={testing_full}/64 full_us={pass_full_us} cached_us={pass_cached_us} cached_gate_evals={pass_gate_evals} cached_child_reads={pass_child_reads} paired_exact=true",
            hits(full.output(), target)
        );
    }
    println!(
        "PAIRED_TRAIN_SUMMARY seed={seed} width={width} proposals={} accepted={accepted_total} full_init_us={full_init_us} cached_init_us={cached_init_us} full_total_us={total_full_us} cached_total_us={total_cached_us} ratio={:.6} outputs_exact=true energy_not_measured=true",
        passes * proposals,
        total_full_us as f64 / total_cached_us as f64
    );
    Ok(())
}

fn power_marker(label: &str) -> Result<(), String> {
    println!("V888_TRAIN_MARKER={label}");
    io::stdout().flush().map_err(|e| e.to_string())
}

/// Dedicated physical host-observer protocol. Build synthetic data and
/// teacher BEFORE the observed windows. Each phase starts from identical
/// parameters and evaluates identical proposals. Energy includes model
/// initialization, acceptance/rebuilds and the complete TRAIN loop, but
/// excludes teacher generation and VALID/TEST scoring (same in both arms).
fn energy_experiment(width: usize, seed: u64) -> Result<(), String> {
    let tree = WideFiveTree::new(width).map_err(|e| format!("{e:?}"))?;
    let train = words(width, seed ^ 0x1111_2222_3333_4444);
    let valid = words(width, seed ^ 0x1357_9bdf_2468_ace0);
    let test = words(width, seed ^ 0x0f0f_f0f0_5a5a_a5a5);
    let options = choices();
    let mut teacher = WidePlasticCircuit::new(&tree, &train).map_err(|e| format!("{e:?}"))?;
    teacher.install_models(|layer, neuron| {
        let bucket = hash(layer, neuron, seed) % 10;
        if layer == 0 {
            options[match bucket {
                0 => 2,
                1 => 3,
                2..=5 => 0,
                _ => 1,
            }]
        } else if bucket % 2 == 0 {
            options[1]
        } else {
            options[4]
        }
    });
    let target = teacher.output();
    let valid_target = teacher
        .evaluate_trace(&valid)
        .map_err(|e| format!("{e:?}"))?
        .last()
        .unwrap()[0];
    let test_target = teacher
        .evaluate_trace(&test)
        .map_err(|e| format!("{e:?}"))?
        .last()
        .unwrap()[0];
    drop(teacher);
    if [target, valid_target, test_target]
        .iter()
        .any(|x| !(16..=48).contains(&x.count_ones()))
    {
        return Err("unbalanced labels in power experiment".into());
    }
    let mut rng = Rng(seed ^ 0xfeed_ba12_8888_0001);
    let proposals: Vec<_> = (0..PASS_COUNT * PROPOSALS_PER_PASS)
        .map(|_| {
            let layer = rng.next() as usize % tree.depth();
            let neuron = rng.next() as usize % tree.level_widths()[layer];
            let model = options[rng.next() as usize % options.len()];
            (layer, neuron, model)
        })
        .collect();
    let order = if seed & 1 == 0 {
        ["FULL", "CACHED"]
    } else {
        ["CACHED", "FULL"]
    };
    let mut expected: Option<(Vec<(u64, bool)>, u32, u32)> = None;
    for arm in order {
        let phase = format!("S{seed}_{arm}");
        power_marker(&format!("{phase}_BEGIN"))?;
        let started = Instant::now();
        let mut circuit = WidePlasticCircuit::new(&tree, &train).map_err(|e| format!("{e:?}"))?;
        let mut decisions = Vec::with_capacity(proposals.len());
        let mut accepted = 0;
        for &(layer, neuron, model) in &proposals {
            let current = circuit.model(layer, neuron).map_err(|e| format!("{e:?}"))?;
            let changed = if model == current {
                false
            } else if arm == "FULL" {
                full_attempt(&mut circuit, layer, neuron, model, target)?.1
            } else {
                cached_attempt(&mut circuit, layer, neuron, model, target)?.1
            };
            accepted += usize::from(changed);
            decisions.push((circuit.output(), changed));
        }
        let elapsed_us = started.elapsed().as_micros();
        power_marker(&format!("{phase}_END"))?;
        let valid_score = valid_hits(&circuit, &valid, valid_target)?;
        let test_score = valid_hits(&circuit, &test, test_target)?;
        let train_score = hits(circuit.output(), target);
        if let Some((previous, prior_valid, prior_test)) = &expected {
            if &decisions != previous || valid_score != *prior_valid || test_score != *prior_test {
                return Err("full/cache physical energy arms diverged".into());
            }
        } else {
            expected = Some((decisions, valid_score, test_score));
        }
        println!(
            "V888_TRAIN_ENERGY_RESULT phase={phase} seed={seed} arm={arm}              proposals={} accepted={accepted} train={train_score}/64              valid={valid_score}/64 test={test_score}/64 elapsed_us={elapsed_us}              matched_all_decisions=true",
            proposals.len()
        );
    }
    Ok(())
}

fn main() -> Result<(), String> {
    if std::env::var_os("V888_TRAIN_POWER_OBSERVER").is_some() {
        for seed in SEEDS {
            energy_experiment(SIX_MILLION_INPUTS, seed)?;
        }
        println!("V888_TRAIN_ENERGY_COMPLETE=true");
        return Ok(());
    }
    for seed in SEEDS {
        experiment(SIX_MILLION_INPUTS, seed, PASS_COUNT, PROPOSALS_PER_PASS)?;
    }
    println!("PAIRED_FULL_VS_CACHED_TRAIN_COMPLETE");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_full_and_cached_training_follow_identical_decisions() {
        experiment(125, SEEDS[0], 2, 32).unwrap();
    }
}
