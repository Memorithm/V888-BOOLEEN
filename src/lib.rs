pub mod activity_atlas;
pub mod activity_guided;
pub mod circuit_explorer;
pub mod connectome;
pub mod consolidated_graded;
pub mod consolidation;
pub mod consolidation_scale;
pub mod eligibility_trace;
pub mod engine;
pub mod fourier5;
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
pub mod fourier5_persistent_session;
pub mod fourier5_population_eligibility;
pub mod fourier5_recurrent;
pub mod fourier5_routing_search;
pub mod fourier5_sequential_adaptive;
pub mod fourier5_shared_context;
pub mod fourier5_task_memory;
pub mod fourier5_wide;
pub mod fourier5_wide_learning;
pub mod full_synapse;
pub mod gates;
pub mod generalization_bench;
pub mod graded_reward;
pub mod importance_protection;
pub mod learning_bench;
pub mod learning_strategies;
pub mod nt_gates;
pub mod online_recall;
pub mod plasticity;
pub mod recurrent_csr;
pub mod region_activation;
pub mod reward_limits;
pub mod source_holdout;
pub mod subcircuit_recall;
pub mod synapse_pairs;

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
pub mod paired_recall_tradeoff;
pub mod region_recall_monitor;
pub mod regional_retention;
pub mod regional_episodes;
pub mod regional_transition_profile;
pub mod regional_lag;
