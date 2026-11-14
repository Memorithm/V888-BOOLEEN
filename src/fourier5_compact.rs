//! Allocation-free, bounded evaluation of a frozen five-way Boolean hierarchy.
//! This is a synthetic circuit representation, not anatomical connectivity.
use crate::fourier5_hierarchy::{FiveCircleHierarchy, GateKind, HierarchyError, MAX_HIERARCHY_INPUTS};
use crate::BooleanState;

/// Evaluate a fixed tree in a single reusable stack buffer. No heap allocation
/// is performed by this function. Each routed leaf is read exactly once and
/// every internal node reads exactly five predecessor bits.
pub fn infer_compact(
    tree: &FiveCircleHierarchy,
    modes: &[GateKind],
    input: &BooleanState,
) -> Result<bool, HierarchyError> {
    if modes.len() != tree.depth() {
        return Err(HierarchyError::InvalidLayerShape);
    }
    if input.len() != tree.input_bits() {
        return Err(HierarchyError::InvalidInput);
    }
    let mut buffer = [false; MAX_HIERARCHY_INPUTS];
    let mut width = tree.input_bits();
    for (position, &source) in tree.routing().iter().enumerate() {
        buffer[position] = input.get(usize::from(source)).map_err(|_| HierarchyError::InvalidRouting)?;
    }
    for &mode in modes {
        let next_width = width / 5;
        for node in 0..next_width {
            let mut bits = 0_u8;
            for bit in 0..5 {
                bits |= u8::from(buffer[node * 5 + bit]) << bit;
            }
            buffer[node] = mode.oracle(bits);
        }
        width = next_width;
    }
    Ok(buffer[0])
}

/// Logical circuit reads, excluding caller-owned input storage and CPU caches.
/// Reports counts, not measured latency or physical memory traffic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompactReadBudget {
    pub leaf_reads: usize,
    pub internal_input_reads: usize,
    pub stack_buffer_bytes: usize,
}

#[must_use]
pub fn read_budget(tree: &FiveCircleHierarchy) -> CompactReadBudget {
    CompactReadBudget {
        leaf_reads: tree.input_bits(),
        internal_input_reads: tree.directed_contacts(),
        stack_buffer_bytes: std::mem::size_of::<[bool; MAX_HIERARCHY_INPUTS]>(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fourier5_hierarchy::LeafRouting;

    #[test]
    fn compact_matches_independent_five_circle_engine_across_depths_and_routes() {
        for depth in 1..=4 {
            let modes = [GateKind::Majority, GateKind::Parity, GateKind::Or, GateKind::And];
            for routing in [LeafRouting::Contiguous, LeafRouting::Interleaved, LeafRouting::SeededShuffle(27)] {
                let tree = FiveCircleHierarchy::from_modes(&modes[..depth], routing).unwrap();
                for seed in 0..16_usize {
                    let mut input = BooleanState::new(tree.input_bits());
                    for bit in 0..tree.input_bits() {
                        input.set(bit, ((bit * 17 + seed * 31) % 11) < 5).unwrap();
                    }
                    assert_eq!(infer_compact(&tree, &modes[..depth], &input), tree.infer_scalar(&input));
                    assert_eq!(infer_compact(&tree, &modes[..depth], &input), tree.infer_lut_scalar(&input));
                }
                let budget = read_budget(&tree);
                assert_eq!(budget.leaf_reads, tree.input_bits());
                assert_eq!(budget.internal_input_reads, tree.neuron_count() * 5);
                assert_eq!(budget.stack_buffer_bytes, 625);
            }
        }
    }

    #[test]
    fn rejects_mismatched_shape_and_input() {
        let modes = [GateKind::Parity, GateKind::And];
        let tree = FiveCircleHierarchy::from_modes(&modes, LeafRouting::Contiguous).unwrap();
        let state = BooleanState::new(25);
        assert_eq!(infer_compact(&tree, &modes[..1], &state), Err(HierarchyError::InvalidLayerShape));
        assert_eq!(infer_compact(&tree, &modes, &BooleanState::new(24)), Err(HierarchyError::InvalidInput));
    }
}
