pub mod ablation_robust;
pub mod activity_atlas;
pub mod activity_guided;
pub mod circuit_explorer;
pub mod connectome;
pub mod consolidated_graded;
pub mod consolidation;
pub mod consolidation_guard;
pub mod consolidation_scale;
pub mod context_cue;
pub mod cue_capacity;
pub mod cue_coverage;
pub mod cue_defer;
pub mod cue_displacement;
pub mod cue_dose_density;
pub mod cue_dose_guard;
pub mod cue_dose_transfer;
pub mod cue_fanin;
pub mod cue_gating;
pub mod cue_init;
pub mod cue_ownership;
pub mod cue_partner;
pub mod cue_reach;
pub mod cue_robustness;
pub mod cue_similarity;
pub mod cue_write_clock;
pub mod cue_write_dose;
pub mod eligibility_trace;
pub mod engine;
pub mod error_gated;
pub mod event_resources;
pub mod event_runtime;
pub mod fourier5;
pub mod fourier5_atomic_session;
pub mod fourier5_conservative_policy;
pub mod fourier5_context_search;
pub mod fourier5_counterfactual_credit;
pub mod fourier5_dynamic;
pub mod fourier5_eligibility;
pub mod fourier5_hidden_credit;
pub mod fourier5_hidden_teacher;
pub mod fourier5_hierarchy;
pub mod fourier5_incremental;
pub mod fourier5_joint_search;
pub mod fourier5_local_learning;
pub mod fourier5_mask_learning;
pub mod fourier5_measured_dispatch;
pub mod fourier5_persistent_session;
pub mod fourier5_physical_probe;
pub mod fourier5_population_eligibility;
pub mod fourier5_recurrent;
pub mod fourier5_routing_search;
pub mod fourier5_scale_policy;
pub mod fourier5_sequential_adaptive;
pub mod fourier5_sequential_phases;
pub mod fourier5_sequential_profile;
pub mod fourier5_shared_context;
pub mod fourier5_task_memory;
pub mod fourier5_valid_geometry_gate;
pub mod fourier5_verified_dispatch;
pub mod fourier5_wide;
pub mod fourier5_wide_learning;
pub mod full_synapse;
pub mod gates;
pub mod generalization_bench;
pub mod gf4_linear;
pub mod graded_reward;
pub mod heldout_ablation;
pub mod importance_protection;
pub mod learning_bench;
pub mod learning_strategies;
pub mod novel_split;
pub mod nt_gates;
pub mod online_recall;
pub mod packed_q2;
pub mod paired_recovery;
pub mod plasticity;
pub mod recovery_profile;
pub mod recurrent_csr;
pub mod region_activation;
pub mod reward_limits;
pub mod seq_generalisation;
pub mod source_holdout;
pub mod subcircuit_recall;
pub mod synapse_pairs;
pub mod task_sequence;

// Brique d'état booléen. Ne constitue pas une simulation du cerveau.

/// Un bit d'activation par position, stocké dans des mots de 64 bits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BooleanState {
    len: usize,
    words: Vec<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateError {
    OutOfBounds,
}

impl BooleanState {
    /// Initialise un état nul de taille donnée.
    #[must_use]
    pub fn new(len: usize) -> Self {
        let full = len / 64;
        let partial = usize::from(len % 64 != 0);
        Self {
            len,
            words: vec![0; full + partial],
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn set(&mut self, index: usize, active: bool) -> Result<(), StateError> {
        if index >= self.len {
            return Err(StateError::OutOfBounds);
        }
        let mask = 1_u64 << (index % 64);
        let word = &mut self.words[index / 64];
        if active {
            *word |= mask;
        } else {
            *word &= !mask;
        }
        Ok(())
    }

    pub fn get(&self, index: usize) -> Result<bool, StateError> {
        if index >= self.len {
            return Err(StateError::OutOfBounds);
        }
        Ok(self.words[index / 64] & (1_u64 << (index % 64)) != 0)
    }

    #[must_use]
    pub fn active_count(&self) -> u64 {
        self.words
            .iter()
            .map(|word| u64::from(word.count_ones()))
            .sum()
    }

    /// Traverse uniquement les indices des bits actifs, par ordre croissant.
    ///
    /// Une présentation à un seul neurone actif ne nécessite donc plus
    /// d'examiner individuellement 188 508 neurones pendant la propagation.
    /// Le parcours reste borné aux mots présents et n'alloue aucun vecteur.
    pub fn active_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.words
            .iter()
            .enumerate()
            .flat_map(|(word_index, &word)| {
                let mut remaining = word;
                std::iter::from_fn(move || {
                    if remaining == 0 {
                        None
                    } else {
                        let bit = remaining.trailing_zeros() as usize;
                        remaining &= remaining - 1;
                        Some(word_index * 64 + bit)
                    }
                })
            })
            .filter(move |&index| index < self.len)
    }

    /// Vue en lecture seule des mots, bit 0 = position 0.
    #[must_use]
    pub fn words(&self) -> &[u64] {
        &self.words
    }
}

#[cfg(test)]
mod tests {
    use super::{BooleanState, StateError};

    #[test]
    fn empty_state() {
        let state = BooleanState::new(0);
        assert!(state.is_empty());
        assert_eq!(state.active_count(), 0);
        assert_eq!(state.get(0), Err(StateError::OutOfBounds));
    }

    #[test]
    fn crosses_word_boundaries() {
        let mut state = BooleanState::new(130);
        for index in [0, 63, 64, 65, 127, 128, 129] {
            state.set(index, true).unwrap();
        }
        assert_eq!(state.active_count(), 7);
        for index in [0, 63, 64, 65, 127, 128, 129] {
            assert_eq!(state.get(index), Ok(true));
        }
        assert_eq!(state.get(130), Err(StateError::OutOfBounds));
        assert_eq!(state.set(130, true), Err(StateError::OutOfBounds));
        state.set(64, false).unwrap();
        assert_eq!(state.active_count(), 6);
        assert_eq!(state.get(64), Ok(false));
    }

    #[test]
    fn clears_and_resets_all_bits() {
        let mut state = BooleanState::new(65);
        for index in 0..65 {
            state.set(index, true).unwrap();
        }
        assert_eq!(state.active_count(), 65);
        assert_eq!(state.words()[0], u64::MAX);
        assert_eq!(state.words()[1], 1);
        for index in 0..65 {
            state.set(index, false).unwrap();
        }
        assert_eq!(state, BooleanState::new(65));
    }

    #[test]
    fn active_bit_indices_are_ordered_across_word_boundaries() {
        let mut state = BooleanState::new(130);
        assert_eq!(
            state.active_indices().collect::<Vec<_>>(),
            Vec::<usize>::new()
        );
        for index in [129, 1, 128, 64, 63, 0, 65, 127] {
            state.set(index, true).unwrap();
        }
        assert_eq!(
            state.active_indices().collect::<Vec<_>>(),
            vec![0, 1, 63, 64, 65, 127, 128, 129]
        );
        state.set(64, false).unwrap();
        assert_eq!(
            state.active_indices().collect::<Vec<_>>(),
            vec![0, 1, 63, 65, 127, 128, 129]
        );
        assert_eq!(state.active_count(), 7);
    }

    #[test]
    fn active_bit_iterator_equals_reference_scan_for_every_length() {
        for len in [0, 1, 63, 64, 65, 127, 128, 129, 130, 188_508] {
            let mut state = BooleanState::new(len);
            for i in 0..len {
                if i % 17 == 3 || i % 113 == 0 {
                    state.set(i, true).unwrap();
                }
            }
            let expected: Vec<usize> = (0..len)
                .filter(|&index| state.get(index).unwrap())
                .collect();
            assert_eq!(state.active_indices().collect::<Vec<_>>(), expected);
        }
    }

    #[test]
    fn padding_bits_remain_zero() {
        let mut state = BooleanState::new(65);
        state.set(64, true).unwrap();
        assert_eq!(state.words()[1], 1);
    }
}
pub mod durable_recovery;
pub mod episode_recall;
pub mod episode_robust_selection;
pub mod fourier5_compact;
pub mod fourier5_joint_audit;
pub mod fourier5_online_policy;
pub mod fourier5_recent_policy;
pub mod fourier5_runtime_breaker;
pub mod fourier5_sequential_accounting;
pub mod fourier5_validation_selection;
pub mod gate_return_recall;
pub mod heldout_regional_recall;
pub mod paired_forgetting;
pub mod paired_recall_tradeoff;
pub mod region_gated_recall;
pub mod region_lag;
pub mod region_persistence;
pub mod region_recall_monitor;
pub mod region_transitions;
pub mod regional_candidate;
pub mod regional_episodes;
pub mod regional_lag;
pub mod regional_retention;
pub mod regional_transition_profile;
pub mod regional_window_consistency;
pub mod regional_window_stability;
pub mod write_allocation;
pub mod write_budget;

pub mod fourier5_verified_replay;

/// Held-out sequential performance and decision-regression audit.
pub mod fourier5_sequential_holdout;

/// Split-disjoint provenance and sequential revision validation for synthetic traces.
pub mod fourier5_trace_provenance;

/// Transactional sequential cache commits checked against an independent Full oracle.
pub mod fourier5_transactional_oracle;

/// Atomic multi-step sequential commits with independent Full oracle checks.
pub mod fourier5_atomic_batch;

/// VALID-only sequential dispatch including measured cache rebuild costs.
pub mod fourier5_sequential_cost_policy;

/// Paired physical timings and cache-epoch safety for synthetic sequential interventions.
pub mod fourier5_sequential_physical_step;

/// Provenance guard against synthetic VALID and holdout sample reuse.
pub mod fourier5_split_guard;

/// Transactional cache epoch admission for verified sequential synthetic interventions.
pub mod fourier5_cache_epoch_guard;

/// Measured transactional sequential cache rebuild with Full-oracle rollback.
pub mod fourier5_sequential_refresh;

/// Transactional delta cache refresh, independently checked against Full.
pub mod fourier5_delta_refresh;

/// Split-isolated measured accounting for sequential cache refresh.
pub mod fourier5_delta_physical_audit;

/// Conservative VALID-calibrated Full/Incremental dispatch.
pub mod fourier5_valid_dispatch;

/// Paired heterogeneous sequential state/cache audit with VmHWM reporting.
pub mod fourier5_sequential_workload_audit;

/// Full-authoritative recovery of sequential synthetic cache state after mismatch.
pub mod fourier5_sequential_full_recovery;

/// Independent VALID-group sequential throughput admission.
pub mod fourier5_sequential_throughput;

/// Transactional sequential cache recovery checked by independent Full/rebuild oracles.
pub mod fourier5_sequential_transaction;

/// VALID-only admission using production-path timings, never verified oracle totals.
pub mod fourier5_production_cost_admission;

/// Fail-closed runtime cone dispatch bound to exact state/cache epochs.
pub mod fourier5_cone_epoch_dispatch;

/// One-way fail-closed breaker for paired sequential synthetic measurements.
pub mod fourier5_sequential_breaker;

/// Split-disjoint measured sequential cohort accounting (synthetic workloads).
pub mod fourier5_sequential_cohort_audit;
