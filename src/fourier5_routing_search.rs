//! Deterministic, train-only selection of *frozen* five-circle leaf connections.
//!
//! All candidates have the same Fourier neurons, depth, five-way fan-in and
//! contact count. Only a bijection of existing leaf sources can change.
//! This is bounded topology search, NOT synaptic plasticity or BANC anatomy.
use std::collections::BTreeSet;

use crate::fourier5_hierarchy::{FiveCircleHierarchy, HierarchyError, BITSLICE_LANES};
use crate::BooleanState;

pub const MAX_SWAP_CANDIDATES: usize = 64;
pub const MAX_ROUTING_TRAIN: usize = 1_024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RoutingSelectionError {
    InvalidCandidateBudget,
    InvalidTraining,
    DuplicateTrainingInput,
    InvalidWidth,
    CandidateBudgetExhausted,
    Hierarchy(HierarchyError),
}

impl From<HierarchyError> for RoutingSelectionError {
    fn from(value: HierarchyError) -> Self {
        Self::Hierarchy(value)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoutingSample {
    pub state: BooleanState,
    pub label: bool,
}

#[derive(Clone, Debug)]
pub struct RoutingSelectionReport {
    pub model: FiveCircleHierarchy,
    /// 0 = unchanged base; 1..N = corresponding candidate swap.
    pub selected_index: usize,
    pub selected_pair: Option<[usize; 2]>,
    pub candidates_evaluated: usize,
    pub train_correct: usize,
    pub train_examples: usize,
    /// Number of full-model inferences on training examples.
    pub training_inferences: usize,
}

#[derive(Clone, Copy, Debug)]
struct Xorshift64(u64);

impl Xorshift64 {
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

/// A proposal bank is constructed ONLY from width and an independent seed.
/// Each swap crosses a five-input group boundary; duplicates are rejected.
/// This does not inspect labels, input patterns or teacher/source metadata.
pub fn candidate_swaps(
    input_bits: usize,
    seed: u64,
    limit: usize,
) -> Result<Vec<[usize; 2]>, RoutingSelectionError> {
    if limit == 0 || limit > MAX_SWAP_CANDIDATES {
        return Err(RoutingSelectionError::InvalidCandidateBudget);
    }
    if !matches!(input_bits, 25 | 125 | 625) {
        return Err(RoutingSelectionError::InvalidWidth);
    }
    let mut rng = Xorshift64::new(seed);
    let mut swaps = Vec::with_capacity(limit);
    // All proposed leaf swaps cross a top-level five-way subtree boundary;
    // the 25-bit case alone has 250 such pairs, enough for 64 candidates.
    for _ in 0..(limit * 256) {
        if swaps.len() == limit {
            break;
        }
        let mut a = (rng.next() % input_bits as u64) as usize;
        let mut b = (rng.next() % input_bits as u64) as usize;
        if a / (input_bits / 5) == b / (input_bits / 5) {
            continue;
        }
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }
        let pair = [a, b];
        if !swaps.contains(&pair) {
            swaps.push(pair);
        }
    }
    if swaps.len() != limit {
        return Err(RoutingSelectionError::CandidateBudgetExhausted);
    }
    Ok(swaps)
}

/// Build one candidate routing from a data-independent swap bank.
/// Index 0 is the original topology. The model parameters are unchanged.
pub fn candidate_model(
    base: &FiveCircleHierarchy,
    swaps: &[[usize; 2]],
    index: usize,
) -> Result<FiveCircleHierarchy, RoutingSelectionError> {
    if index > swaps.len() {
        return Err(RoutingSelectionError::InvalidCandidateBudget);
    }
    if index == 0 {
        return Ok(base.clone());
    }
    let [a, b] = swaps[index - 1];
    if a >= base.input_bits() || b >= base.input_bits() || a == b {
        return Err(RoutingSelectionError::InvalidCandidateBudget);
    }
    let mut routing = base.routing().to_vec();
    routing.swap(a, b);
    Ok(base.rerouted(routing)?)
}

#[derive(Clone, Debug)]
struct PackedTrain {
    planes: Vec<u64>,
    labels: u64,
}

fn pack_train(
    model: &FiveCircleHierarchy,
    train: &[RoutingSample],
) -> Result<Vec<PackedTrain>, RoutingSelectionError> {
    if train.is_empty()
        || train.len() > MAX_ROUTING_TRAIN
        || train.len() % BITSLICE_LANES != 0
    {
        return Err(RoutingSelectionError::InvalidTraining);
    }
    let mut unique = BTreeSet::new();
    for row in train {
        if row.state.len() != model.input_bits() {
            return Err(RoutingSelectionError::InvalidWidth);
        }
        if !unique.insert(row.state.words().to_vec()) {
            return Err(RoutingSelectionError::DuplicateTrainingInput);
        }
    }
    let mut groups = Vec::with_capacity(train.len() / BITSLICE_LANES);
    for chunk in train.chunks_exact(BITSLICE_LANES) {
        let states: Vec<_> = chunk.iter().map(|row| row.state.clone()).collect();
        let planes = FiveCircleHierarchy::transpose64(&states)?;
        let mut labels = 0_u64;
        for (lane, row) in chunk.iter().enumerate() {
            labels |= u64::from(row.label) << lane;
        }
        groups.push(PackedTrain { planes, labels });
    }
    Ok(groups)
}

fn correct(
    model: &FiveCircleHierarchy,
    groups: &[PackedTrain],
) -> Result<usize, RoutingSelectionError> {
    let mut correct = 0;
    for group in groups {
        let predictions = model.infer_bitplanes(&group.planes)?;
        correct += (BITSLICE_LANES as u32 - (predictions ^ group.labels).count_ones()) as usize;
    }
    Ok(correct)
}

/// Exhaustively score the predeclared bank on TRAIN *only*, with the exact
/// same bit-sliced evaluator and sample budget for each routing candidate.
/// Ties keep the lower candidate index, making selection reproducible.
/// Validation, TEST and shifted-distribution labels are never accepted here.
pub fn select_route(
    base: &FiveCircleHierarchy,
    train: &[RoutingSample],
    proposal_seed: u64,
    swap_budget: usize,
) -> Result<RoutingSelectionReport, RoutingSelectionError> {
    let groups = pack_train(base, train)?;
    let swaps = candidate_swaps(base.input_bits(), proposal_seed, swap_budget)?;
    let mut winner = base.clone();
    let mut best_hits = correct(base, &groups)?;
    let mut best_index = 0;
    for index in 1..=swaps.len() {
        let trial = candidate_model(base, &swaps, index)?;
        let hits = correct(&trial, &groups)?;
        if hits > best_hits {
            winner = trial;
            best_hits = hits;
            best_index = index;
        }
    }
    Ok(RoutingSelectionReport {
        model: winner,
        selected_index: best_index,
        selected_pair: if best_index == 0 {
            None
        } else {
            Some(swaps[best_index - 1])
        },
        candidates_evaluated: swaps.len() + 1,
        train_correct: best_hits,
        train_examples: train.len(),
        training_inferences: (swaps.len() + 1) * train.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fourier5_hierarchy::{GateKind, LeafRouting};

    fn make_train(
        teacher: &FiveCircleHierarchy,
        size: usize,
        seed: u64,
    ) -> Vec<RoutingSample> {
        let mut rng = Xorshift64::new(seed);
        let mut seen = BTreeSet::new();
        let mut data = Vec::with_capacity(size);
        while data.len() < size {
            let mut state = BooleanState::new(teacher.input_bits());
            for bit in 0..state.len() {
                if rng.next() & 1 != 0 {
                    state.set(bit, true).unwrap();
                }
            }
            if !seen.insert(state.words().to_vec()) {
                continue;
            }
            let label = teacher.infer_scalar(&state).unwrap();
            data.push(RoutingSample { state, label });
        }
        data
    }

    #[test]
    fn candidate_bank_is_deterministic_and_crosses_five_input_groups() {
        for width in [25, 125, 625] {
            let first = candidate_swaps(width, 19, 64).unwrap();
            let second = candidate_swaps(width, 19, 64).unwrap();
            assert_eq!(first, second);
            assert_eq!(first.len(), 64);
            let mut seen = BTreeSet::new();
            for [a, b] in first {
                assert!(a < b && b < width);
                assert_ne!(a / (width / 5), b / (width / 5));
                assert!(seen.insert((a, b)));
            }
        }
    }

    #[test]
    fn select_one_swap_from_train_without_changing_five_circle_parameters() {
        let modes = [GateKind::Majority, GateKind::Parity];
        let base = FiveCircleHierarchy::from_modes(&modes, LeafRouting::Contiguous).unwrap();
        let swaps = candidate_swaps(25, 73, 32).unwrap();
        let teacher = candidate_model(&base, &swaps, 11).unwrap();
        let train = make_train(&teacher, 512, 0x79_4432);
        let fit = select_route(&base, &train, 73, 32).unwrap();
        assert_eq!(fit.train_correct, train.len());
        assert_eq!(fit.candidates_evaluated, 33);
        assert_eq!(fit.training_inferences, 33 * train.len());
        assert_eq!(fit.model.neuron_count(), base.neuron_count());
        assert_eq!(fit.model.directed_contacts(), base.directed_contacts());
        let heldout = make_train(&teacher, 256, 0x85_1321);
        assert!(
            heldout
                .iter()
                .all(|row| fit.model.infer_scalar(&row.state) == Ok(row.label))
        );
        let second = select_route(&base, &train, 73, 32).unwrap();
        assert_eq!(fit.selected_index, second.selected_index);
        assert_eq!(fit.model.routing(), second.model.routing());
    }

    #[test]
    fn rerouting_preserves_compiled_tables_and_rejects_bad_permutations() {
        let base = FiveCircleHierarchy::from_modes(
            &[GateKind::Majority, GateKind::Parity, GateKind::Parity],
            LeafRouting::Contiguous,
        )
        .unwrap();
        let mut route = base.routing().to_vec();
        route.swap(0, 13);
        let rerouted = base.rerouted(route).unwrap();
        assert_eq!(rerouted.neuron_count(), base.neuron_count());
        assert_eq!(rerouted.compiled_truth_bytes(), base.compiled_truth_bytes());
        assert!(matches!(
            base.rerouted(vec![0; 125]),
            Err(HierarchyError::InvalidRouting)
        ));
        assert!(matches!(
            base.rerouted(vec![0; 124]),
            Err(HierarchyError::InvalidRouting)
        ));
        assert!(matches!(
            base.rerouted((1..=125).collect()),
            Err(HierarchyError::InvalidRouting)
        ));
    }

    #[test]
    fn refuses_training_duplicates_invalid_width_and_unbounded_search() {
        let base = FiveCircleHierarchy::from_modes(
            &[GateKind::Majority, GateKind::Parity],
            LeafRouting::Contiguous,
        )
        .unwrap();
        let train = make_train(&base, 64, 1);
        assert!(matches!(
            select_route(&base, &[], 1, 8),
            Err(RoutingSelectionError::InvalidTraining)
        ));
        assert!(matches!(
            select_route(&base, &train[..63], 1, 8),
            Err(RoutingSelectionError::InvalidTraining)
        ));
        let mut duplicate = train.clone();
        duplicate[1] = duplicate[0].clone();
        assert!(matches!(
            select_route(&base, &duplicate, 1, 8),
            Err(RoutingSelectionError::DuplicateTrainingInput)
        ));
        let mut wrong_width = train.clone();
        wrong_width[1].state = BooleanState::new(24);
        assert!(matches!(
            select_route(&base, &wrong_width, 1, 8),
            Err(RoutingSelectionError::InvalidWidth)
        ));
        assert!(matches!(
            select_route(&base, &train, 1, 0),
            Err(RoutingSelectionError::InvalidCandidateBudget)
        ));
        assert!(matches!(
            select_route(&base, &train, 1, MAX_SWAP_CANDIDATES + 1),
            Err(RoutingSelectionError::InvalidCandidateBudget)
        ));
        assert!(matches!(
            select_route(&base, &vec![train[0].clone(); MAX_ROUTING_TRAIN + 1], 1, 8),
            Err(RoutingSelectionError::InvalidTraining)
        ));
    }
}
