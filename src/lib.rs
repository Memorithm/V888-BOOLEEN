pub mod connectome;
pub mod engine;
pub mod full_synapse;
pub mod gates;
pub mod learning_bench;
pub mod source_holdout;
pub mod nt_gates;
pub mod plasticity;
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
    fn padding_bits_remain_zero() {
        let mut state = BooleanState::new(65);
        state.set(64, true).unwrap();
        assert_eq!(state.words()[1], 1);
    }
}
