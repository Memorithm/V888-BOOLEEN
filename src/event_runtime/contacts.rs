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

/// Immutable, source-indexed cached reductions. Only worthwhile sources
/// are retained; a missing key selects the normal CSR reference path.
pub(super) type ContactCache = BTreeMap<usize, Vec<(usize, u128)>>;

/// Construction statistics, not measured saved joules or memory traffic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ContactCacheReport {
    /// Number of distinct sources retained (never more than pair budget).
    pub cached_sources: usize,
    /// Total retained (source,target) weighted pairs.
    pub cached_pairs: usize,
    /// Logical raw CSR contact rows covered by retained sources.
    pub covered_contact_rows: usize,
    /// Sources with enough outgoing rows to be considered for compression.
    pub candidate_sources: usize,
}

/// Build a bounded retained cache outside event execution.
///
/// Only sources with >=64 CSR rows and at least 4x reduction in the number
/// of distinct nonzero targets are included. No source is partially cached.
/// max_pairs=0 selects the uncached path. Retained pairs are bounded, but
/// temporary aggregation during construction can allocate more.
pub(super) fn build_contact_cache(
    graph: &Connectome,
    max_pairs: usize,
) -> Result<(Option<ContactCache>, ContactCacheReport), EventError> {
    let mut cache = ContactCache::new();
    let mut report = ContactCacheReport::default();
    if max_pairs == 0 {
        return Ok((None, report));
    }
    for source in 0..graph.ids.len() {
        let first = graph.offsets[source] as usize;
        let last = graph.offsets[source + 1] as usize;
        let fanout = last - first;
        if fanout < 64 {
            continue;
        }
        report.candidate_sources += 1;
        let pairs = aggregate_outgoing(graph, source)?;
        if pairs.is_empty() || pairs.len().saturating_mul(4) > fanout {
            continue;
        }
        if pairs.len() > max_pairs - report.cached_pairs {
            continue;
        }
        report.cached_pairs += pairs.len();
        report.cached_sources += 1;
        report.covered_contact_rows += fanout;
        cache.insert(source, pairs);
    }
    if cache.is_empty() {
        Ok((None, report))
    } else {
        Ok((Some(cache), report))
    }
}

/// Get a cached reduction when present, otherwise fall back to exact CSR
/// reduction. Parallel preflight owns a Vec so it clones only compressed
/// weighted destination pairs, never the original large CSR run.
pub(super) fn outgoing_with_cache(
    graph: &Connectome,
    cache: Option<&ContactCache>,
    source: usize,
) -> Result<Vec<(usize, u128)>, EventError> {
    match cache.and_then(|stored| stored.get(&source)) {
        Some(deltas) => Ok(deltas.clone()),
        None => aggregate_outgoing(graph, source),
    }
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
            counts.push(if i % 11 == 0 {
                0
            } else {
                (i as u64) * 1_000_000
            });
        }
        let graph = graph(targets, counts);
        assert_eq!(
            aggregate_outgoing(&graph, 0).unwrap(),
            independent_reference(&graph)
        );
    }

    #[test]
    fn all_three_event_engines_match_on_mixed_sorted_and_unsorted_target_runs() {
        use super::super::EventRuntime;

        let graph = Connectome {
            ids: vec![10, 20, 30, 40],
            offsets: vec![0, 8, 11, 11, 11],
            targets: vec![2, 2, 3, 2, 3, 2, 3, 2, 2, 2, 3],
            counts: vec![1, 0, 3, 2, 0, 1, 4, 2, 2, 1, 1],
        };
        graph.validate().unwrap();
        let mut serial = EventRuntime::new(&graph, 3, 24).unwrap();
        let mut frontier = EventRuntime::new(&graph, 3, 24).unwrap();
        let mut disjoint = EventRuntime::new(&graph, 3, 24).unwrap();
        for engine in [&mut serial, &mut frontier, &mut disjoint] {
            engine.schedule_drive(0, 10, true).unwrap();
            engine.schedule_drive(0, 20, true).unwrap();
            engine.schedule_drive(3, 10, false).unwrap();
            engine.schedule_drive(3, 20, false).unwrap();
        }
        let expected = serial.run(48).unwrap();
        assert_eq!(
            frontier.run_parallel_frontiers(48, 4).unwrap().work,
            expected
        );
        assert_eq!(
            disjoint.run_disjoint_parallel_commits(48, 4).unwrap().work,
            expected
        );
        for engine in [&frontier, &disjoint] {
            assert_eq!(engine.state(), serial.state());
            assert_eq!(engine.incoming, serial.incoming);
            assert_eq!(engine.pending, serial.pending);
            assert_eq!(engine.last_processed_time(), serial.last_processed_time());
            assert_eq!(engine.audit_incoming(), Ok(true));
        }
        assert_eq!(serial.audit_incoming(), Ok(true));
        assert_eq!(serial.pending_count(), 0);
    }

    #[test]
    fn zero_weight_middle_of_a_run_never_inserts_a_spurious_target() {
        let graph = graph(vec![1, 0, 1, 2], vec![3, 0, 4, 0]);
        assert_eq!(aggregate_outgoing(&graph, 0), Ok(vec![(1, 7)]));
    }
}
