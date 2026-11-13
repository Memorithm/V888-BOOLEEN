//! Output-label-only counterfactual credit assignment on Boolean Fourier trees.
//!
//! This module never receives a teacher network or hidden target activations.
//! Each proposal changes one hidden-neuron signed Walsh coefficient/bias,
//! or swaps two existing leaf sources. The *actual final output loss on
//! TRAIN* decides whether to commit an intervention. All models retain
//! the same neuron count, contact count and five Walsh-mask identities.
//! Greedy and blind (label-independent) controls use equal proposal budgets.
use std::collections::BTreeSet;

use crate::fourier5::{Circle, FiveCircle, CIRCLES};
use crate::fourier5_hierarchy::{FiveCircleHierarchy, HierarchyError, BITSLICE_LANES};
use crate::fourier5_incremental::{FiveCircleIncrementalCache, PackedIntervention};
use crate::fourier5_routing_search::{RoutingSample, MAX_ROUTING_TRAIN};

pub const MAX_CREDIT_ROUNDS: usize = 16;
pub const MAX_CREDIT_PROPOSALS_PER_ROUND: usize = 128;
pub const MAX_CREDIT_WEIGHT: i32 = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CreditScope {
    HiddenParameters,
    OutputParameters,
    LeafRouting,
    JointHiddenAndRouting,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CreditPolicy {
    /// Commit only a strict TRAIN improvement; no teacher taps.
    Greedy,
    /// Blindly choose an intervention before seeing its scored outputs.
    /// Every proposed candidate is still evaluated for a matched compute budget.
    Blind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CreditConfig {
    pub scope: CreditScope,
    pub policy: CreditPolicy,
    pub rounds: usize,
    pub proposals_per_round: usize,
    pub seed: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CreditError {
    InvalidBudget,
    InvalidInput,
    DuplicateInput,
    Hierarchy(HierarchyError),
}

impl From<HierarchyError> for CreditError {
    fn from(value: HierarchyError) -> Self {
        Self::Hierarchy(value)
    }
}

#[derive(Clone, Debug)]
pub struct CreditResult {
    pub model: FiveCircleHierarchy,
    pub initial_train_correct: usize,
    pub final_train_correct: usize,
    pub train_examples: usize,
    pub candidate_evaluations: usize,
    pub full_model_train_inferences: usize,
    pub accepted_parameter_moves: usize,
    pub accepted_routing_moves: usize,
    pub rounds_without_improvement: usize,
    pub rounds_run: usize,
    /// Gate calls needed to build all immutable 64-lane round baselines,
    /// including the initial complete evaluation.
    pub cache_build_gate_evaluations: usize,
    /// Exact gate calls used to evaluate only counterfactual dependency cones.
    pub delta_gate_evaluations: usize,
    /// Gate calls that full-model reevaluation of every candidate would need.
    /// Counts 64-lane gates, NOT wall time or scalar Boolean predictions.
    pub full_recompute_gate_evaluations: usize,
}

#[derive(Clone, Copy)]
struct Xorshift(u64);
impl Xorshift {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9e37_79b9_7f4a_7c15
        } else {
            seed
        })
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

#[derive(Clone)]
struct Packed {
    planes: Vec<u64>,
    labels: u64,
}

fn pack(base: &FiveCircleHierarchy, train: &[RoutingSample]) -> Result<Vec<Packed>, CreditError> {
    if train.len() < BITSLICE_LANES
        || train.len() > MAX_ROUTING_TRAIN
        || train.len() % BITSLICE_LANES != 0
    {
        return Err(CreditError::InvalidInput);
    }
    let mut seen = BTreeSet::new();
    let mut batches = Vec::with_capacity(train.len() / BITSLICE_LANES);
    for row in train {
        if row.state.len() != base.input_bits() {
            return Err(CreditError::InvalidInput);
        }
        if !seen.insert(row.state.words().to_vec()) {
            return Err(CreditError::DuplicateInput);
        }
    }
    for chunk in train.chunks_exact(BITSLICE_LANES) {
        let states: Vec<_> = chunk.iter().map(|r| r.state.clone()).collect();
        let planes = FiveCircleHierarchy::transpose64(&states)?;
        let mut labels = 0_u64;
        for (lane, row) in chunk.iter().enumerate() {
            labels |= u64::from(row.label) << lane;
        }
        batches.push(Packed { planes, labels });
    }
    Ok(batches)
}

fn correct(model: &FiveCircleHierarchy, batches: &[Packed]) -> Result<usize, CreditError> {
    let mut count = 0;
    for batch in batches {
        let result = model.infer_bitplanes(&batch.planes)?;
        count += (BITSLICE_LANES as u32 - (result ^ batch.labels).count_ones()) as usize;
    }
    Ok(count)
}

fn signed(circle: Circle) -> i32 {
    let radius = circle.radius.min(MAX_CREDIT_WEIGHT as u32) as i32;
    if circle.phase_pi {
        -radius
    } else {
        radius
    }
}

/// One parameter move only: no mask replacements, no hidden supervision,
/// no new neuron, no new source. Radius/phase is a signed Walsh weight.
fn perturb_parameter(
    model: &FiveCircleHierarchy,
    scope: CreditScope,
    random: &mut Xorshift,
) -> Result<(FiveCircleHierarchy, PackedIntervention), CreditError> {
    let depth = model.depth();
    let allowed = if scope == CreditScope::OutputParameters {
        1
    } else {
        model.neuron_count() - 1
    };
    if allowed == 0 {
        return Err(CreditError::InvalidBudget);
    }
    let mut offset = (random.next() % allowed as u64) as usize;
    let (layer, node) = if scope == CreditScope::OutputParameters {
        (depth - 1, 0)
    } else {
        let mut selected = None;
        for layer in 0..depth - 1 {
            let width = model.input_bits() / CIRCLES.pow((layer + 1) as u32);
            if offset < width {
                selected = Some((layer, offset));
                break;
            }
            offset -= width;
        }
        selected.expect("hidden neuron count matches five-way layers")
    };

    let coordinate = (random.next() % (CIRCLES + 1) as u64) as usize;
    let steps = [-4_i32, -2, -1, 1, 2, 4];
    let delta = steps[(random.next() % steps.len() as u64) as usize];
    let current = model.layer_model(layer, node)?;
    let mut circles = *current.circles();
    let mut bias = current.bias();
    if coordinate == CIRCLES {
        bias = (bias + delta).clamp(-MAX_CREDIT_WEIGHT, MAX_CREDIT_WEIGHT);
    } else {
        let value =
            (signed(circles[coordinate]) + delta).clamp(-MAX_CREDIT_WEIGHT, MAX_CREDIT_WEIGHT);
        circles[coordinate].radius = value.unsigned_abs();
        circles[coordinate].phase_pi = value < 0;
    }
    let next = FiveCircle::new(circles, bias).expect("unchanged valid Walsh masks");
    Ok((
        model.with_neuron_parameters(layer, node, next)?,
        PackedIntervention::Parameter { layer, neuron: node, model: next },
    ))
}

/// Explicit top-level cross-group swap: every source remains used exactly
/// once. Proposals depend on fixed seed and current routing, not the labels.
fn perturb_route(
    model: &FiveCircleHierarchy,
    random: &mut Xorshift,
) -> Result<(FiveCircleHierarchy, PackedIntervention), CreditError> {
    let width = model.input_bits();
    let stride = width / CIRCLES;
    let a = (random.next() % width as u64) as usize;
    let group = a / stride;
    let other_group = (group + 1 + (random.next() % (CIRCLES - 1) as u64) as usize) % CIRCLES;
    let b = other_group * stride + (random.next() % stride as u64) as usize;
    let mut routing = model.routing().to_vec();
    routing.swap(a, b);
    Ok((
        model.rerouted(routing)?,
        PackedIntervention::LeafSwap { first: a, second: b },
    ))
}

/// Select among interventions by measuring exact OUTPUT LABEL error on
/// TRAIN only, never reading a teacher's hidden outputs or held-out labels.
/// Each scope and policy consumes the SAME number of candidate/full-network
/// evaluations (1 + rounds * proposals), even across plateaus. No inference
/// state is changed in place; interventions are applied to fresh clones.
pub fn fit_final_loss(
    base: &FiveCircleHierarchy,
    train: &[RoutingSample],
    config: CreditConfig,
) -> Result<CreditResult, CreditError> {
    if base.depth() < 2
        || !(1..=MAX_CREDIT_ROUNDS).contains(&config.rounds)
        || !(1..=MAX_CREDIT_PROPOSALS_PER_ROUND).contains(&config.proposals_per_round)
    {
        return Err(CreditError::InvalidBudget);
    }
    let batches = pack(base, train)?;
    let initial = correct(base, &batches)?;
    let mut model = base.clone();
    let mut hits = initial;
    let mut rng = Xorshift::new(config.seed);
    let mut accepted_parameter_moves = 0;
    let mut accepted_routing_moves = 0;
    let mut rounds_without_improvement = 0;
    let mut cache_build_gate_evaluations = batches.len() * base.neuron_count();
    let mut delta_gate_evaluations = 0;

    for _ in 0..config.rounds {
        // One full 64-lane baseline traversal per batch per round.
        // Interventions then evaluate *only* gates inside changed cones.
        let caches: Vec<_> = batches
            .iter()
            .map(|batch| FiveCircleIncrementalCache::new(&model, &batch.planes))
            .collect::<Result<_, _>>()?;
        cache_build_gate_evaluations += caches
            .iter()
            .map(FiveCircleIncrementalCache::build_gate_evaluations)
            .sum::<usize>();
        debug_assert_eq!(
            caches.iter().zip(&batches).map(|(cache, batch)| {
                (BITSLICE_LANES as u32 - (cache.baseline_output() ^ batch.labels).count_ones()) as usize
            }).sum::<usize>(),
            hits
        );
        let blind_index = (rng.next() % config.proposals_per_round as u64) as usize;
        let mut winning_model: Option<FiveCircleHierarchy> = None;
        let mut winning_hits = hits;
        let mut winning_is_route = false;
        for index in 0..config.proposals_per_round {
            let route = match config.scope {
                CreditScope::LeafRouting => true,
                CreditScope::JointHiddenAndRouting => rng.next() & 3 == 0,
                CreditScope::HiddenParameters | CreditScope::OutputParameters => false,
            };
            let (proposal, delta) = if route {
                perturb_route(&model, &mut rng)?
            } else {
                perturb_parameter(&model, config.scope, &mut rng)?
            };
            let mut candidate_hits = 0;
            for (cache, batch) in caches.iter().zip(&batches) {
                let evaluation = cache.evaluate(delta)?;
                delta_gate_evaluations += evaluation.gate_evaluations;
                candidate_hits += (BITSLICE_LANES as u32
                    - (evaluation.output ^ batch.labels).count_ones()) as usize;
            }
            let chosen = match config.policy {
                CreditPolicy::Greedy => candidate_hits > winning_hits,
                CreditPolicy::Blind => index == blind_index,
            };
            if chosen {
                winning_model = Some(proposal);
                winning_hits = candidate_hits;
                winning_is_route = route;
            }
        }
        if let Some(next) = winning_model {
            if winning_hits <= hits {
                rounds_without_improvement += 1;
            }
            hits = winning_hits;
            model = next;
            if winning_is_route {
                accepted_routing_moves += 1;
            } else {
                accepted_parameter_moves += 1;
            }
        } else {
            rounds_without_improvement += 1;
        }
    }
    let evaluations = config.rounds * config.proposals_per_round;
    Ok(CreditResult {
        model,
        initial_train_correct: initial,
        final_train_correct: hits,
        train_examples: train.len(),
        candidate_evaluations: evaluations,
        full_model_train_inferences: (evaluations + 1) * train.len(),
        accepted_parameter_moves,
        accepted_routing_moves,
        rounds_without_improvement,
        rounds_run: config.rounds,
        cache_build_gate_evaluations,
        delta_gate_evaluations,
        full_recompute_gate_evaluations: (evaluations + 1)
            * batches.len()
            * base.neuron_count(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fourier5_hierarchy::{GateKind, LeafRouting};
    use crate::BooleanState;

    fn sample(teacher: &FiveCircleHierarchy, count: usize) -> Vec<RoutingSample> {
        let mut rng = Xorshift::new(0xf381_75a1);
        let mut seen = BTreeSet::new();
        let mut rows = Vec::new();
        while rows.len() < count {
            let mut state = BooleanState::new(teacher.input_bits());
            for i in 0..state.len() {
                if rng.next() & 0x10000 != 0 {
                    state.set(i, true).unwrap();
                }
            }
            if !seen.insert(state.words().to_vec()) {
                continue;
            }
            let label = teacher.infer_scalar(&state).unwrap();
            rows.push(RoutingSample { state, label });
        }
        rows
    }

    fn config(scope: CreditScope, policy: CreditPolicy) -> CreditConfig {
        CreditConfig {
            scope,
            policy,
            rounds: 4,
            proposals_per_round: 64,
            seed: 0x35_1e_10,
        }
    }

    #[test]
    fn equal_budget_and_monotone_training_without_hidden_teacher_labels() {
        let teacher = FiveCircleHierarchy::from_modes(
            &[GateKind::Majority, GateKind::Parity],
            LeafRouting::Contiguous,
        )
        .unwrap();
        let mut student = teacher.clone();
        student = student
            .with_neuron_parameters(0, 0, GateKind::And.five_circle())
            .unwrap();
        let train = sample(&teacher, 256);
        let greedy = fit_final_loss(
            &student,
            &train,
            config(CreditScope::HiddenParameters, CreditPolicy::Greedy),
        )
        .unwrap();
        let blind = fit_final_loss(
            &student,
            &train,
            config(CreditScope::HiddenParameters, CreditPolicy::Blind),
        )
        .unwrap();
        assert!(greedy.final_train_correct >= greedy.initial_train_correct);
        assert_eq!(
            greedy.full_model_train_inferences,
            blind.full_model_train_inferences
        );
        assert_eq!(greedy.candidate_evaluations, 256);
        assert_eq!(greedy.model.routing(), student.routing());
        assert_eq!(
            greedy.model.directed_contacts(),
            student.directed_contacts()
        );
        let copy = fit_final_loss(
            &student,
            &train,
            config(CreditScope::HiddenParameters, CreditPolicy::Greedy),
        )
        .unwrap();
        assert_eq!(greedy.final_train_correct, copy.final_train_correct);
        for l in 0..student.depth() {
            let count = if l == 0 { 5 } else { 1 };
            for n in 0..count {
                let original = student.layer_model(l, n).unwrap();
                let changed = greedy.model.layer_model(l, n).unwrap();
                for i in 0..CIRCLES {
                    assert_eq!(original.circles()[i].mask, changed.circles()[i].mask);
                }
            }
        }
    }

    #[test]
    fn routing_never_changes_neurons_and_output_scope_never_changes_hidden() {
        let base = FiveCircleHierarchy::from_modes(
            &[GateKind::And, GateKind::Parity],
            LeafRouting::Contiguous,
        )
        .unwrap();
        let teacher = FiveCircleHierarchy::from_modes(
            &[GateKind::Majority, GateKind::Parity],
            LeafRouting::Contiguous,
        )
        .unwrap();
        let train = sample(&teacher, 128);
        let route = fit_final_loss(
            &base,
            &train,
            config(CreditScope::LeafRouting, CreditPolicy::Greedy),
        )
        .unwrap();
        for l in 0..2 {
            for i in 0..if l == 0 { 5 } else { 1 } {
                assert_eq!(route.model.layer_model(l, i), base.layer_model(l, i));
            }
        }
        let output = fit_final_loss(
            &base,
            &train,
            config(CreditScope::OutputParameters, CreditPolicy::Greedy),
        )
        .unwrap();
        assert_eq!(output.model.routing(), base.routing());
        for i in 0..5 {
            assert_eq!(output.model.layer_model(0, i), base.layer_model(0, i));
        }
    }

    #[test]
    fn rejects_bad_budget_duplicate_and_wrong_width_before_mutation() {
        let base = FiveCircleHierarchy::from_modes(
            &[GateKind::Majority, GateKind::Parity],
            LeafRouting::Contiguous,
        )
        .unwrap();
        let train = sample(&base, 128);
        let mut invalid = config(CreditScope::JointHiddenAndRouting, CreditPolicy::Greedy);
        invalid.rounds = 0;
        assert!(matches!(
            fit_final_loss(&base, &train, invalid),
            Err(CreditError::InvalidBudget)
        ));
        let good = config(CreditScope::JointHiddenAndRouting, CreditPolicy::Greedy);
        assert!(matches!(
            fit_final_loss(&base, &train[..63], good),
            Err(CreditError::InvalidInput)
        ));
        let mut dup = train.clone();
        dup[1] = dup[0].clone();
        assert!(matches!(
            fit_final_loss(&base, &dup, good),
            Err(CreditError::DuplicateInput)
        ));
        let mut wrong = train.clone();
        wrong[1].state = BooleanState::new(24);
        assert!(matches!(
            fit_final_loss(&base, &wrong, good),
            Err(CreditError::InvalidInput)
        ));
        invalid = good;
        invalid.proposals_per_round = MAX_CREDIT_PROPOSALS_PER_ROUND + 1;
        assert!(matches!(
            fit_final_loss(&base, &train, invalid),
            Err(CreditError::InvalidBudget)
        ));
    }
}
