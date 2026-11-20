//! Fail-closed VALID geometry coverage gate for measured dispatch.
//! TRAIN, TEST and OOD observations must never enter this admission policy.

/// A geometry is identified by the total gate count and affected cone size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub gates: u64,
    pub cone: u64,
}

/// A paired VALID observation, including rejected oracle comparisons.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValidPair {
    pub geometry: Geometry,
    pub full_ns: u64,
    pub incremental_refresh_ns: u64,
    pub oracle_equal: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineChoice {
    Full,
    Incremental,
}

/// Require independent geometries, exact oracle agreement and an integer
/// saving margin. No floating point rounding, no acceptance on overflow.
/// The caller must enforce the disjoint VALID provenance contract.
#[must_use]
pub fn choose_for_geometry(
    valid: &[ValidPair],
    target: Geometry,
    minimum_distinct: usize,
    minimum_samples: usize,
    minimum_saving_percent: u8,
) -> EngineChoice {
    if minimum_distinct < 2
        || minimum_samples < minimum_distinct
        || minimum_saving_percent > 100
        || target.gates == 0
        || target.cone > target.gates
    {
        return EngineChoice::Full;
    }
    let mut distinct = Vec::new();
    let mut full = 0_u128;
    let mut incremental = 0_u128;
    let mut samples = 0_usize;
    for pair in valid {
        if pair.geometry.gates == 0
            || pair.geometry.cone > pair.geometry.gates
            || !pair.oracle_equal
        {
            return EngineChoice::Full;
        }
        // Never extrapolate across total gate counts or beyond measured cones.
        if pair.geometry.gates != target.gates || pair.geometry.cone < target.cone {
            continue;
        }
        // A pooled speedup can hide a slow cone. Reject any measured
        // geometry whose incremental refresh is not strictly faster.
        if pair.full_ns == 0 || pair.incremental_refresh_ns >= pair.full_ns {
            return EngineChoice::Full;
        }
        samples = match samples.checked_add(1) {
            Some(value) => value,
            None => return EngineChoice::Full,
        };
        full = match full.checked_add(u128::from(pair.full_ns)) {
            Some(value) => value,
            None => return EngineChoice::Full,
        };
        incremental = match incremental.checked_add(u128::from(pair.incremental_refresh_ns)) {
            Some(value) => value,
            None => return EngineChoice::Full,
        };
        if !distinct.contains(&pair.geometry) {
            distinct.push(pair.geometry);
        }
    }
    if samples < minimum_samples || distinct.len() < minimum_distinct || full == 0 {
        return EngineChoice::Full;
    }
    let lhs = incremental.checked_mul(100);
    let rhs = full.checked_mul(u128::from(100 - minimum_saving_percent));
    if matches!((lhs, rhs), (Some(a), Some(b)) if a < b) {
        EngineChoice::Incremental
    } else {
        EngineChoice::Full
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(cone: u64, incremental: u64) -> ValidPair {
        ValidPair {
            geometry: Geometry { gates: 100, cone },
            full_ns: 100,
            incremental_refresh_ns: incremental,
            oracle_equal: true,
        }
    }

    #[test]
    fn requires_distinct_covered_geometries() {
        let target = Geometry { gates: 100, cone: 10 };
        assert_eq!(
            choose_for_geometry(&[pair(12, 10), pair(12, 10)], target, 2, 2, 20),
            EngineChoice::Full
        );
        assert_eq!(
            choose_for_geometry(&[pair(12, 10), pair(20, 10)], target, 2, 2, 20),
            EngineChoice::Incremental
        );
    }

    #[test]
    fn never_extrapolates_to_larger_cone_or_different_gate_count() {
        let valid = [pair(12, 10), pair(20, 10)];
        assert_eq!(
            choose_for_geometry(&valid, Geometry { gates: 100, cone: 21 }, 2, 2, 20),
            EngineChoice::Full
        );
        assert_eq!(
            choose_for_geometry(&valid, Geometry { gates: 101, cone: 10 }, 2, 2, 20),
            EngineChoice::Full
        );
    }

    #[test]
    fn oracle_mismatch_and_cost_regression_fail_closed() {
        let target = Geometry { gates: 100, cone: 10 };
        let mut valid = [pair(12, 10), pair(20, 10)];
        valid[0].oracle_equal = false;
        assert_eq!(choose_for_geometry(&valid, target, 2, 2, 20), EngineChoice::Full);
        valid[0].oracle_equal = true;
        valid[0].incremental_refresh_ns = 300;
        assert_eq!(choose_for_geometry(&valid, target, 2, 2, 20), EngineChoice::Full);
    }

    #[test]
    fn pooled_gain_never_masks_one_slow_covered_cone() {
        let target = Geometry { gates: 100, cone: 10 };
        let valid = [pair(12, 1), pair(20, 101)];
        // Aggregate refresh is 102 ns versus 200 ns Full, but the
        // second cone regresses. Dispatch must remain Full.
        assert_eq!(choose_for_geometry(&valid, target, 2, 2, 20), EngineChoice::Full);
    }

    #[test]
    fn threshold_is_strict_and_invalid_policy_is_rejected() {
        let target = Geometry { gates: 100, cone: 10 };
        let valid = [pair(12, 80), pair(20, 80)];
        assert_eq!(choose_for_geometry(&valid, target, 2, 2, 20), EngineChoice::Full);
        assert_eq!(choose_for_geometry(&valid, target, 2, 2, 19), EngineChoice::Incremental);
        assert_eq!(choose_for_geometry(&valid, target, 1, 2, 19), EngineChoice::Full);
    }
}

/// Split assignment is explicit: only VALID pairs may calibrate dispatch.
/// Stable sample IDs must be unique across all partitions and within VALID.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Split {
    Train,
    Valid,
    Test,
    Ood,
}

/// The ID is a stable identity of an underlying intervention/workload,
/// not a position in a shuffled dataset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplitPair {
    pub sample_id: u64,
    pub split: Split,
    pub pair: ValidPair,
}

/// Fail closed if any sample identity appears twice, including cross-split
/// leakage or repeated VALID measurements mislabeled as independent samples.
/// The admission policy sees only VALID pairs; TEST/OOD never tune thresholds.
#[must_use]
pub fn choose_with_disjoint_splits(
    observations: &[SplitPair],
    target: Geometry,
    minimum_distinct: usize,
    minimum_samples: usize,
    minimum_saving_percent: u8,
) -> EngineChoice {
    let mut ids = std::collections::HashSet::with_capacity(observations.len());
    let mut valid = Vec::new();
    for observation in observations {
        if !ids.insert(observation.sample_id) {
            return EngineChoice::Full;
        }
        if observation.split == Split::Valid {
            valid.push(observation.pair);
        }
    }
    // This API has sample IDs but no workload-group provenance. Independent
    // groups must be checked by choose_with_disjoint_groups instead.
    choose_for_geometry(
        &valid,
        target,
        minimum_distinct,
        minimum_samples,
        minimum_saving_percent,
    )
}

#[cfg(test)]
mod disjoint_tests {
    use super::*;

    fn observation(id: u64, split: Split, cone: u64) -> SplitPair {
        SplitPair {
            sample_id: id,
            split,
            pair: ValidPair {
                geometry: Geometry { gates: 100, cone },
                full_ns: 100,
                incremental_refresh_ns: 20,
                oracle_equal: true,
            },
        }
    }

    #[test]
    fn disjoint_valid_only_matches_independent_oracle() {
        let target = Geometry { gates: 100, cone: 10 };
        let data = [
            observation(1, Split::Train, 12),
            observation(2, Split::Valid, 12),
            observation(3, Split::Valid, 20),
            observation(4, Split::Test, 12),
            observation(5, Split::Ood, 20),
        ];
        let expected = choose_for_geometry(
            &[data[1].pair, data[2].pair],
            target,
            2,
            2,
            20,
        );
        assert_eq!(expected, EngineChoice::Incremental);
        assert_eq!(
            choose_with_disjoint_splits(&data, target, 2, 2, 20),
            expected
        );
    }

    #[test]
    fn cross_split_identity_leakage_fails_closed() {
        let target = Geometry { gates: 100, cone: 10 };
        let data = [
            observation(7, Split::Train, 12),
            observation(7, Split::Valid, 12),
            observation(8, Split::Valid, 20),
        ];
        assert_eq!(
            choose_with_disjoint_splits(&data, target, 2, 2, 20),
            EngineChoice::Full
        );
    }

    #[test]
    fn duplicated_valid_identity_is_not_independent_evidence() {
        let target = Geometry { gates: 100, cone: 10 };
        let data = [
            observation(7, Split::Valid, 12),
            observation(7, Split::Valid, 20),
        ];
        assert_eq!(
            choose_with_disjoint_splits(&data, target, 2, 2, 20),
            EngineChoice::Full
        );
    }

    #[test]
    fn train_test_ood_cannot_supply_missing_valid_coverage() {
        let target = Geometry { gates: 100, cone: 10 };
        let data = [
            observation(1, Split::Valid, 12),
            observation(2, Split::Train, 20),
            observation(3, Split::Test, 20),
            observation(4, Split::Ood, 20),
        ];
        assert_eq!(
            choose_with_disjoint_splits(&data, target, 2, 2, 20),
            EngineChoice::Full
        );
    }
}

/// Group-level holdout: correlated interventions must never cross splits.
/// A group may contain multiple distinct samples within one split.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupedSplitPair {
    pub group_id: u64,
    pub observation: SplitPair,
}

/// Reject correlated cross-partition workloads before calibrating on VALID.
/// Group identifiers must derive from original workloads, not split positions.
#[must_use]
pub fn choose_with_disjoint_groups(
    observations: &[GroupedSplitPair],
    target: Geometry,
    minimum_distinct: usize,
    minimum_samples: usize,
    minimum_saving_percent: u8,
) -> EngineChoice {
    let mut groups = std::collections::HashMap::new();
    let mut ids = std::collections::HashSet::new();
    let mut valid = Vec::new();
    let mut valid_groups = std::collections::HashSet::new();
    for grouped in observations {
        let observation = grouped.observation;
        if !ids.insert(observation.sample_id) {
            return EngineChoice::Full;
        }
        match groups.insert(grouped.group_id, observation.split) {
            Some(previous) if previous != observation.split => return EngineChoice::Full,
            _ => {}
        }
        if observation.split == Split::Valid {
            valid_groups.insert(grouped.group_id);
            valid.push(observation.pair);
        }
    }
    // Geometry diversity is not evidence of independent workload coverage.
    if valid_groups.len() < minimum_distinct {
        return EngineChoice::Full;
    }
    choose_for_geometry(
        &valid,
        target,
        minimum_distinct,
        minimum_samples,
        minimum_saving_percent,
    )
}

#[cfg(test)]
mod grouped_tests {
    use super::*;

    fn sample(id: u64, group: u64, split: Split, cone: u64) -> GroupedSplitPair {
        GroupedSplitPair {
            group_id: group,
            observation: SplitPair {
                sample_id: id,
                split,
                pair: ValidPair {
                    geometry: Geometry { gates: 100, cone },
                    full_ns: 100,
                    incremental_refresh_ns: 20,
                    oracle_equal: true,
                },
            },
        }
    }

    #[test]
    fn correlated_cross_split_samples_fail_closed() {
        let data = [
            sample(1, 42, Split::Train, 12),
            sample(2, 42, Split::Valid, 12),
            sample(3, 43, Split::Valid, 20),
        ];
        assert_eq!(
            choose_with_disjoint_groups(&data, Geometry { gates: 100, cone: 10 }, 2, 2, 20),
            EngineChoice::Full
        );
    }

    #[test]
    fn distinct_groups_allow_valid_dispatch() {
        let data = [
            sample(1, 41, Split::Train, 12),
            sample(2, 42, Split::Valid, 12),
            sample(3, 44, Split::Valid, 20),
            sample(4, 43, Split::Test, 12),
        ];
        assert_eq!(
            choose_with_disjoint_groups(&data, Geometry { gates: 100, cone: 10 }, 2, 2, 20),
            EngineChoice::Incremental
        );
    }

    #[test]
    fn correlated_valid_geometries_do_not_count_as_independent_groups() {
        let data = [
            sample(1, 42, Split::Valid, 12),
            sample(2, 42, Split::Valid, 20),
        ];
        // Independent scalar oracle: one group cannot satisfy a two-group gate.
        let distinct_groups: std::collections::HashSet<_> =
            data.iter().map(|row| row.group_id).collect();
        assert_eq!(distinct_groups.len(), 1);
        assert_eq!(
            choose_with_disjoint_groups(&data, Geometry { gates: 100, cone: 10 }, 2, 2, 20),
            EngineChoice::Full
        );
    }

    #[test]
    fn duplicate_sample_in_same_group_fails_closed() {
        let data = [
            sample(2, 42, Split::Valid, 12),
            sample(2, 42, Split::Valid, 20),
        ];
        assert_eq!(
            choose_with_disjoint_groups(&data, Geometry { gates: 100, cone: 10 }, 2, 2, 20),
            EngineChoice::Full
        );
    }
}
