//! Exact CSR outgoing-contact reduction with a contiguous-target fast path.
//!
//! The ordinary ordered map is updated once per maximal nonzero target run,
//! instead of once per edge. Arbitrary (unsorted) CSR targets remain valid;
//! the result is sorted and the u128 weighted sums are unchanged. Every CSR
//! row is still examined, including rows whose weight is zero.
//!
//! This is an algorithmic implementation, not a claim of physical speedup.

use std::collections::BTreeMap;

use super::EventError;
use crate::connectome::Connectome;

/// Return nonzero aggregated outgoing contributions in ascending target order.
/// An empty result cannot schedule any wakeups, but still consumes the source
/// event and counts its complete CSR range in EventWork.
pub(super) fn aggregate_outgoing(
    graph: &Connectome,
    source: usize,
) -> Result<Vec<(usize, u128)>, EventError> {
    let start = graph.offsets[source] as usize;
    let end = graph.offsets[source + 1] as usize;
    let mut totals = BTreeMap::<usize, u128>::new();
    let mut previous = None;
    let mut consecutive_sum = 0_u128;

    // The pending consecutive run is flushed only when its destination
    // changes. This avoids O(edges) repeated BTreeMap lookups on CSR runs
    // with repeated target IDs (a common case in the synthetic wide probes).
    for edge in start..end {
        let weight = u128::from(graph.counts[edge]);
        if weight == 0 {
            continue;
        }
        let target = graph.targets[edge] as usize;
        if previous == Some(target) {
            consecutive_sum = consecutive_sum
                .checked_add(weight)
                .ok_or(EventError::ArithmeticOverflow)?;
            continue;
        }
        if let Some(old_target) = previous {
            let entry = totals.entry(old_target).or_insert(0);
            *entry = entry
                .checked_add(consecutive_sum)
                .ok_or(EventError::ArithmeticOverflow)?;
        }
        previous = Some(target);
        consecutive_sum = weight;
    }
    if let Some(target) = previous {
        let entry = totals.entry(target).or_insert(0);
        *entry = entry
            .checked_add(consecutive_sum)
            .ok_or(EventError::ArithmeticOverflow)?;
    }
    Ok(totals.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph(targets: Vec<u32>, counts: Vec<u64>) -> Connectome {
        assert_eq!(targets.len(), counts.len());
        let len = targets.len() as u64;
        let size = 1 + targets.iter().copied().max().unwrap_or(0) as usize;
        let mut offsets = vec![len; size + 1];
        offsets[0] = 0;
        Connectome {
            ids: (10..10 + size as u64).collect(),
            offsets,
            targets,
            counts,
        }
    }

    fn independent_reference(graph: &Connectome) -> Vec<(usize, u128)> {
        let mut total = BTreeMap::new();
        for (target, count) in graph.targets.iter().zip(&graph.counts) {
            if *count != 0 {
                *total.entry(*target as usize).or_insert(0_u128) += u128::from(*count);
            }
        }
        total.into_iter().collect()
    }

    #[test]
    fn sparse_zero_weight_and_contiguous_runs_match_independent_reference() {
        let testcases = [
            (vec![], vec![]),
            (vec![1, 1, 1, 2, 2, 1], vec![1, 0, 4, 0, 6, 2]),
            (vec![1, 2, 1, 2, 1, 2], vec![1, 2, 3, 4, 5, 6]),
            (vec![0, 0, 0, 0], vec![0, 0, 0, 0]),
            (vec![2, 2, 2], vec![u64::MAX, u64::MAX, u64::MAX]),
        ];
        for (targets, weights) in testcases {
            let graph = graph(targets, weights);
            graph.validate().expect("valid synthetic CSR");
            assert_eq!(
                aggregate_outgoing(&graph, 0).unwrap(),
                independent_reference(&graph)
            );
        }
    }

    #[test]
    fn shuffled_nonmonotone_weighted_contacts_preserve_exact_u128_totals() {
        let mut targets = Vec::new();
        let mut counts = Vec::new();
        for i in 0..768_usize {
            let target = (i * 17 / 7 + i % 5) % 7;
            targets.push(target as u32);
            counts.push(if i % 11 == 0 { 0 } else { (i as u64) * 1_000_000 });
        }
        let graph = graph(targets, counts);
        assert_eq!(aggregate_outgoing(&graph, 0).unwrap(), independent_reference(&graph));
    }

    #[test]
    fn zero_weight_middle_of_a_run_never_inserts_a_spurious_target() {
        let graph = graph(vec![1, 0, 1, 2], vec![3, 0, 4, 0]);
        assert_eq!(aggregate_outgoing(&graph, 0), Ok(vec![(1, 7)]));
    }
}
