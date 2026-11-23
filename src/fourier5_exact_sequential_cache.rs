//! Exact sequential Boolean DAG cache, independent of biological BANC data.
//! A changed input propagates through every affected descendant, including
//! reconvergent paths. No performance claim is made without RemoteOps measures.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gate {
    Input(usize),
    Not(usize),
    And(usize, usize),
    Xor(usize, usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheError {
    InvalidDependency,
    InvalidInput,
}

#[derive(Clone, Debug)]
pub struct ExactSequentialCache {
    gates: Vec<Gate>,
    inputs: Vec<bool>,
    values: Vec<bool>,
    dependents: Vec<Vec<usize>>,
    input_roots: Vec<Vec<usize>>,
}

impl ExactSequentialCache {
    pub fn new(gates: Vec<Gate>, inputs: Vec<bool>) -> Result<Self, CacheError> {
        let mut dependents = vec![Vec::new(); gates.len()];
        let mut input_roots = vec![Vec::new(); inputs.len()];
        for (index, gate) in gates.iter().enumerate() {
            let parents: &[usize] = match gate {
                Gate::Input(input) => {
                    if *input >= inputs.len() {
                        return Err(CacheError::InvalidInput);
                    }
                    input_roots[*input].push(index);
                    &[]
                }
                Gate::Not(parent) => std::slice::from_ref(parent),
                Gate::And(a, b) | Gate::Xor(a, b) => {
                    if *a >= index || *b >= index {
                        return Err(CacheError::InvalidDependency);
                    }
                    dependents[*a].push(index);
                    dependents[*b].push(index);
                    continue;
                }
            };
            for &parent in parents {
                if parent >= index {
                    return Err(CacheError::InvalidDependency);
                }
                dependents[parent].push(index);
            }
        }
        let mut cache = Self {
            values: vec![false; gates.len()],
            gates,
            inputs,
            dependents,
            input_roots,
        };
        cache.rebuild();
        Ok(cache)
    }

    fn evaluate(&self, index: usize) -> bool {
        match self.gates[index] {
            Gate::Input(i) => self.inputs[i],
            Gate::Not(a) => !self.values[a],
            Gate::And(a, b) => self.values[a] & self.values[b],
            Gate::Xor(a, b) => self.values[a] ^ self.values[b],
        }
    }

    /// Full topological rebuild, used as the independent reference path.
    pub fn rebuild(&mut self) {
        for index in 0..self.gates.len() {
            self.values[index] = self.evaluate(index);
        }
    }

    /// Apply one accepted input intervention and refresh its complete causal
    /// cone. Returns number of gates evaluated; unchanged inputs cost zero.
    pub fn update_input(&mut self, input: usize, value: bool) -> Result<usize, CacheError> {
        if input >= self.inputs.len() {
            return Err(CacheError::InvalidInput);
        }
        if self.inputs[input] == value {
            return Ok(0);
        }
        self.inputs[input] = value;
        // A sparse, topologically ordered worklist avoids scanning every gate
        // for a small intervention cone. Each descendant is evaluated once,
        // including reconvergent paths, after all its parents are current.
        let mut pending = std::collections::BTreeSet::new();
        pending.extend(self.input_roots[input].iter().copied());
        let mut evaluated = 0;
        while let Some(index) = pending.pop_first() {
            let next = self.evaluate(index);
            evaluated += 1;
            if next != self.values[index] {
                self.values[index] = next;
                pending.extend(self.dependents[index].iter().copied());
            }
        }
        Ok(evaluated)
    }

    /// Atomically apply a batch of interventions. Validate all indices before
    /// mutating state; repeated indices use their last requested value.
    /// Every affected gate is evaluated at most once in topological order.
    /// Returns the number of evaluated gates, including input roots.
    pub fn update_inputs_atomic(
        &mut self,
        changes: &[(usize, bool)],
    ) -> Result<usize, CacheError> {
        if changes.iter().any(|(index, _)| *index >= self.inputs.len()) {
            return Err(CacheError::InvalidInput);
        }
        let mut final_values = std::collections::BTreeMap::new();
        for &(index, value) in changes {
            final_values.insert(index, value);
        }
        let mut pending = std::collections::BTreeSet::new();
        for (index, value) in final_values {
            if self.inputs[index] != value {
                self.inputs[index] = value;
                pending.extend(self.input_roots[index].iter().copied());
            }
        }
        let mut evaluated = 0;
        while let Some(index) = pending.pop_first() {
            let next = self.evaluate(index);
            evaluated += 1;
            if next != self.values[index] {
                self.values[index] = next;
                pending.extend(self.dependents[index].iter().copied());
            }
        }
        Ok(evaluated)
    }

    /// Checked execution mode for qualification and safety-critical routing.
    /// The independent full topological pass is evaluated after each accepted
    /// batch. If incremental state diverges, replace it with the full result.
    /// This mode deliberately trades speed for an exact, fail-closed decision.
    /// The returned flag reports whether incremental agreed with Full.
    pub fn update_inputs_verified(
        &mut self,
        changes: &[(usize, bool)],
    ) -> Result<(usize, bool), CacheError> {
        // Invalid batches must not change either the inputs or the cache.
        if changes.iter().any(|(index, _)| *index >= self.inputs.len()) {
            return Err(CacheError::InvalidInput);
        }
        let evaluated = self.update_inputs_atomic(changes)?;
        // Compute the reference without consulting cached intermediate values.
        let mut reference = vec![false; self.gates.len()];
        for (index, gate) in self.gates.iter().enumerate() {
            reference[index] = match *gate {
                Gate::Input(input) => self.inputs[input],
                Gate::Not(parent) => !reference[parent],
                Gate::And(a, b) => reference[a] & reference[b],
                Gate::Xor(a, b) => reference[a] ^ reference[b],
            };
        }
        let exact = reference == self.values;
        if !exact {
            self.values = reference;
        }
        Ok((evaluated, exact))
    }

    /// Conservative budget dispatcher: count the full structural cone before
    /// choosing a sparse update. A zero budget always selects Full for changes.
    /// This is a deterministic safety baseline, NOT a measured latency policy.
    /// The return flag is true only when the incremental path was selected.
    pub fn update_inputs_bounded(
        &mut self,
        changes: &[(usize, bool)],
        max_cone_gates: usize,
    ) -> Result<(usize, bool), CacheError> {
        if changes.iter().any(|(index, _)| *index >= self.inputs.len()) {
            return Err(CacheError::InvalidInput);
        }
        let mut final_values = std::collections::BTreeMap::new();
        for &(index, value) in changes {
            final_values.insert(index, value);
        }
        let mut pending = std::collections::BTreeSet::new();
        for (&input, &value) in &final_values {
            if self.inputs[input] != value {
                pending.extend(self.input_roots[input].iter().copied());
            }
        }
        if pending.is_empty() {
            return Ok((0, true));
        }
        // Count all structurally reachable gates, including reconvergence,
        // without relying on value-dependent pruning or mutating state.
        let mut cone = pending.clone();
        while let Some(index) = pending.pop_first() {
            for &child in &self.dependents[index] {
                if cone.insert(child) {
                    pending.insert(child);
                }
            }
            if cone.len() > max_cone_gates {
                break;
            }
        }
        if cone.len() > max_cone_gates {
            for (input, value) in final_values {
                self.inputs[input] = value;
            }
            self.rebuild();
            return Ok((self.gates.len(), false));
        }
        let evaluated = self.update_inputs_atomic(changes)?;
        Ok((evaluated, true))
    }

    pub fn values(&self) -> &[bool] {
        &self.values
    }

    pub fn inputs(&self) -> &[bool] {
        &self.inputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_full_oracle_after_reconvergent_interventions() {
        let gates = vec![
            Gate::Input(0), Gate::Input(1), Gate::Not(0),
            Gate::And(0, 1), Gate::Xor(2, 3),
            Gate::And(4, 1), Gate::Xor(5, 3),
        ];
        let mut incremental = ExactSequentialCache::new(gates.clone(), vec![false, false]).unwrap();
        for (input, value) in [(0, true), (1, true), (0, false), (1, false), (0, true)] {
            let evaluated = incremental.update_input(input, value).unwrap();
            assert!(evaluated > 0 && evaluated <= gates.len());
            // Separate construction and complete evaluation on every step.
            let full = ExactSequentialCache::new(gates.clone(), incremental.inputs().to_vec()).unwrap();
            assert_eq!(incremental.values(), full.values());
        }
    }

    #[test]
    fn rejects_cycles_and_invalid_inputs_without_mutation() {
        assert!(matches!(
            ExactSequentialCache::new(vec![Gate::Not(0)], vec![]),
            Err(CacheError::InvalidDependency)
        ));
        assert!(matches!(
            ExactSequentialCache::new(vec![Gate::Input(1)], vec![false]),
            Err(CacheError::InvalidInput)
        ));
        let mut cache = ExactSequentialCache::new(vec![Gate::Input(0)], vec![false]).unwrap();
        assert_eq!(cache.update_input(1, true), Err(CacheError::InvalidInput));
        assert_eq!(cache.values(), &[false]);
        assert_eq!(cache.update_input(0, false), Ok(0));
    }
    #[test]
    fn sparse_cone_matches_independent_scalar_oracle() {
        let mut gates = Vec::new();
        for i in 0..128 {
            gates.push(Gate::Input(i));
        }
        gates.push(Gate::And(0, 1)); // 128
        gates.push(Gate::Xor(128, 2)); // 129
        gates.push(Gate::Not(128)); // 130, reconvergent
        gates.push(Gate::Xor(129, 130)); // 131
        let mut cache = ExactSequentialCache::new(gates.clone(), vec![false; 128]).unwrap();
        for (input, value) in [(0, true), (2, true), (1, true), (0, false), (127, true)] {
            let visited = cache.update_input(input, value).unwrap();
            assert!(visited <= 5);
            let mut expected = vec![false; gates.len()];
            for (index, gate) in gates.iter().enumerate() {
                expected[index] = match *gate {
                    Gate::Input(i) => cache.inputs()[i],
                    Gate::Not(a) => !expected[a],
                    Gate::And(a, b) => expected[a] && expected[b],
                    Gate::Xor(a, b) => expected[a] ^ expected[b],
                };
            }
            assert_eq!(cache.values(), expected);
        }
    }

    #[test]
    fn unchanged_intermediate_gate_prunes_descendants_without_changing_oracle() {
        let gates = vec![
            Gate::Input(0),
            Gate::Input(1),
            Gate::And(0, 1),
            Gate::Not(2),
            Gate::Xor(3, 2),
        ];
        let mut cache = ExactSequentialCache::new(gates.clone(), vec![false, false]).unwrap();
        // Input 0 changes but AND(false, false) remains false: no descendant work.
        assert_eq!(cache.update_input(0, true), Ok(2));
        assert_eq!(cache.values(), &[true, false, false, true, true]);
        // The second intervention changes AND and propagates to both descendants.
        assert_eq!(cache.update_inputs_atomic(&[(1, true)]), Ok(4));
        let mut oracle = vec![false; gates.len()];
        for (index, gate) in gates.iter().enumerate() {
            oracle[index] = match *gate {
                Gate::Input(i) => cache.inputs()[i],
                Gate::Not(a) => !oracle[a],
                Gate::And(a, b) => oracle[a] && oracle[b],
                Gate::Xor(a, b) => oracle[a] ^ oracle[b],
            };
        }
        assert_eq!(cache.values(), oracle);
    }

    #[test]
    fn atomic_batch_matches_independent_scalar_full_oracle() {
        let gates = vec![
            Gate::Input(0), Gate::Input(1), Gate::Input(2),
            Gate::Xor(0, 1), Gate::And(3, 2), Gate::Not(3),
            Gate::Xor(4, 5), Gate::And(6, 3),
        ];
        let mut cache = ExactSequentialCache::new(gates.clone(), vec![false; 3]).unwrap();
        for changes in [
            vec![(0, true), (1, true), (2, true)],
            vec![(0, false), (0, true), (1, false)],
            vec![(2, false), (0, false)],
            vec![(1, true), (1, false)],
        ] {
            let visited = cache.update_inputs_atomic(&changes).unwrap();
            assert!(visited <= gates.len());
            let mut expected = vec![false; gates.len()];
            for (i, gate) in gates.iter().enumerate() {
                expected[i] = match *gate {
                    Gate::Input(j) => cache.inputs()[j],
                    Gate::Not(a) => !expected[a],
                    Gate::And(a, b) => expected[a] && expected[b],
                    Gate::Xor(a, b) => expected[a] ^ expected[b],
                };
            }
            assert_eq!(cache.values(), expected);
        }
        let previous_inputs = cache.inputs().to_vec();
        let previous_values = cache.values().to_vec();
        assert_eq!(cache.update_inputs_atomic(&[(0, true), (3, true)]), Err(CacheError::InvalidInput));
        assert_eq!(cache.inputs(), previous_inputs);
        assert_eq!(cache.values(), previous_values);
        assert_eq!(cache.update_inputs_atomic(&[]), Ok(0));
    }

    #[test]
    fn verified_sequential_batches_match_independent_full_oracle() {
        let gates = vec![
            Gate::Input(0), Gate::Input(1), Gate::Input(2),
            Gate::Xor(0, 1), Gate::And(3, 2), Gate::Not(3),
            Gate::Xor(4, 5), Gate::And(6, 3),
        ];
        let mut cache = ExactSequentialCache::new(gates.clone(), vec![false; 3]).unwrap();
        for changes in [
            vec![(0, true), (1, true)],
            vec![(2, true)],
            vec![(0, false), (1, false)],
            vec![(2, false), (2, true)],
            vec![(0, true), (0, false)],
        ] {
            let (_, exact) = cache.update_inputs_verified(&changes).unwrap();
            assert!(exact);
            let mut oracle = vec![false; gates.len()];
            for (index, gate) in gates.iter().enumerate() {
                oracle[index] = match *gate {
                    Gate::Input(i) => cache.inputs()[i],
                    Gate::Not(a) => !oracle[a],
                    Gate::And(a, b) => oracle[a] & oracle[b],
                    Gate::Xor(a, b) => oracle[a] ^ oracle[b],
                };
            }
            assert_eq!(cache.values(), oracle);
        }
        let previous = cache.values().to_vec();
        assert_eq!(cache.update_inputs_verified(&[(3, true)]), Err(CacheError::InvalidInput));
        assert_eq!(cache.values(), previous);
    }

    #[test]
    fn bounded_dispatch_matches_scalar_oracle_across_sequential_batches() {
        let gates = vec![
            Gate::Input(0),
            Gate::Input(1),
            Gate::And(0, 1),
            Gate::Not(2),
            Gate::Xor(2, 3),
            Gate::Xor(4, 1),
        ];
        for budget in [0, 1, 2, 3, 6, usize::MAX] {
            let mut cache = ExactSequentialCache::new(gates.clone(), vec![false; 2]).unwrap();
            for changes in [
                vec![(0, true)],
                vec![(1, true)],
                vec![(0, false), (1, false)],
                vec![(0, true), (0, false)],
            ] {
                let (evaluated, incremental) =
                    cache.update_inputs_bounded(&changes, budget).unwrap();
                assert!(evaluated <= gates.len());
                if budget == 0 && evaluated > 0 {
                    assert!(!incremental);
                }
                let mut oracle = vec![false; gates.len()];
                for (i, gate) in gates.iter().enumerate() {
                    oracle[i] = match *gate {
                        Gate::Input(j) => cache.inputs()[j],
                        Gate::Not(a) => !oracle[a],
                        Gate::And(a, b) => oracle[a] & oracle[b],
                        Gate::Xor(a, b) => oracle[a] ^ oracle[b],
                    };
                }
                assert_eq!(cache.values(), oracle);
            }
            let before = cache.values().to_vec();
            assert_eq!(
                cache.update_inputs_bounded(&[(2, true)], budget),
                Err(CacheError::InvalidInput)
            );
            assert_eq!(cache.values(), before);
        }
    }

    #[test]
    fn verified_mode_repairs_stale_cached_state() {
        let mut cache = ExactSequentialCache::new(
            vec![Gate::Input(0), Gate::Not(0), Gate::Xor(0, 1)],
            vec![false],
        ).unwrap();
        cache.values[2] = !cache.values[2];
        let (_, exact) = cache.update_inputs_verified(&[]).unwrap();
        assert!(!exact);
        assert_eq!(cache.values(), &[false, true, true]);
        assert_eq!(cache.update_inputs_verified(&[]), Ok((0, true)));
    }


}
