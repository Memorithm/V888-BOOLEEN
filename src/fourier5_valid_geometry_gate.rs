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
        let target = Geometry {
            gates: 100,
            cone: 10,
        };
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
            choose_for_geometry(
                &valid,
                Geometry {
                    gates: 100,
                    cone: 21
                },
                2,
                2,
                20
            ),
            EngineChoice::Full
        );
        assert_eq!(
            choose_for_geometry(
                &valid,
                Geometry {
                    gates: 101,
                    cone: 10
                },
                2,
                2,
                20
            ),
            EngineChoice::Full
        );
    }

    #[test]
    fn oracle_mismatch_and_cost_regression_fail_closed() {
        let target = Geometry {
            gates: 100,
            cone: 10,
        };
        let mut valid = [pair(12, 10), pair(20, 10)];
        valid[0].oracle_equal = false;
        assert_eq!(
            choose_for_geometry(&valid, target, 2, 2, 20),
            EngineChoice::Full
        );
        valid[0].oracle_equal = true;
        valid[0].incremental_refresh_ns = 300;
        assert_eq!(
            choose_for_geometry(&valid, target, 2, 2, 20),
            EngineChoice::Full
        );
    }

    #[test]
    fn pooled_gain_never_masks_one_slow_covered_cone() {
        let target = Geometry {
            gates: 100,
            cone: 10,
        };
        let valid = [pair(12, 1), pair(20, 101)];
        // Aggregate refresh is 102 ns versus 200 ns Full, but the
        // second cone regresses. Dispatch must remain Full.
        assert_eq!(
            choose_for_geometry(&valid, target, 2, 2, 20),
            EngineChoice::Full
        );
    }

    #[test]
    fn threshold_is_strict_and_invalid_policy_is_rejected() {
        let target = Geometry {
            gates: 100,
            cone: 10,
        };
        let valid = [pair(12, 80), pair(20, 80)];
        assert_eq!(
            choose_for_geometry(&valid, target, 2, 2, 20),
            EngineChoice::Full
        );
        assert_eq!(
            choose_for_geometry(&valid, target, 2, 2, 19),
            EngineChoice::Incremental
        );
        assert_eq!(
            choose_for_geometry(&valid, target, 1, 2, 19),
            EngineChoice::Full
        );
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
        let target = Geometry {
            gates: 100,
            cone: 10,
        };
        let data = [
            observation(1, Split::Train, 12),
            observation(2, Split::Valid, 12),
            observation(3, Split::Valid, 20),
            observation(4, Split::Test, 12),
            observation(5, Split::Ood, 20),
        ];
        let expected = choose_for_geometry(&[data[1].pair, data[2].pair], target, 2, 2, 20);
        assert_eq!(expected, EngineChoice::Incremental);
        assert_eq!(
            choose_with_disjoint_splits(&data, target, 2, 2, 20),
            expected
        );
    }

    #[test]
    fn cross_split_identity_leakage_fails_closed() {
        let target = Geometry {
            gates: 100,
            cone: 10,
        };
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
        let target = Geometry {
            gates: 100,
            cone: 10,
        };
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
        let target = Geometry {
            gates: 100,
            cone: 10,
        };
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
            choose_with_disjoint_groups(
                &data,
                Geometry {
                    gates: 100,
                    cone: 10
                },
                2,
                2,
                20
            ),
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
            choose_with_disjoint_groups(
                &data,
                Geometry {
                    gates: 100,
                    cone: 10
                },
                2,
                2,
                20
            ),
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
            choose_with_disjoint_groups(
                &data,
                Geometry {
                    gates: 100,
                    cone: 10
                },
                2,
                2,
                20
            ),
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
            choose_with_disjoint_groups(
                &data,
                Geometry {
                    gates: 100,
                    cone: 10
                },
                2,
                2,
                20
            ),
            EngineChoice::Full
        );
    }
}

/// Conservative group-balanced admission: every independent VALID workload group
/// must independently meet the requested saving margin on every covered sample.
/// This prevents a large, fast group from masking a smaller, slow group.
/// TEST/OOD never contribute to calibration. No measured speedup is implied.
#[must_use]
pub fn choose_with_group_balanced_margin(
    observations: &[GroupedSplitPair],
    target: Geometry,
    minimum_distinct: usize,
    minimum_samples: usize,
    minimum_saving_percent: u8,
) -> EngineChoice {
    if choose_with_disjoint_groups(
        observations,
        target,
        minimum_distinct,
        minimum_samples,
        minimum_saving_percent,
    ) != EngineChoice::Incremental
    {
        return EngineChoice::Full;
    }
    let mut covered_groups = std::collections::HashSet::new();
    for grouped in observations {
        let row = grouped.observation;
        if row.split != Split::Valid {
            continue;
        }
        let pair = row.pair;
        if pair.geometry.gates != target.gates || pair.geometry.cone < target.cone {
            continue;
        }
        covered_groups.insert(grouped.group_id);
        // Use checked integer arithmetic, strict improvement and no rounding.
        let lhs = u128::from(pair.incremental_refresh_ns).checked_mul(100);
        let rhs = u128::from(pair.full_ns).checked_mul(u128::from(100 - minimum_saving_percent));
        if !matches!((lhs, rhs), (Some(a), Some(b)) if a < b) {
            return EngineChoice::Full;
        }
    }
    if covered_groups.len() < minimum_distinct {
        EngineChoice::Full
    } else {
        EngineChoice::Incremental
    }
}

#[cfg(test)]
mod balanced_margin_tests {
    use super::*;

    fn row(id: u64, group: u64, cone: u64, incremental: u64) -> GroupedSplitPair {
        GroupedSplitPair {
            group_id: group,
            observation: SplitPair {
                sample_id: id,
                split: Split::Valid,
                pair: ValidPair {
                    geometry: Geometry { gates: 100, cone },
                    full_ns: 100,
                    incremental_refresh_ns: incremental,
                    oracle_equal: true,
                },
            },
        }
    }

    #[test]
    fn large_fast_group_cannot_mask_slow_independent_group() {
        let mut data: Vec<_> = (0..20).map(|i| row(i, 1, 12, 1)).collect();
        data.push(row(21, 2, 20, 95));
        let target = Geometry {
            gates: 100,
            cone: 10,
        };
        assert_eq!(
            choose_with_disjoint_groups(&data, target, 2, 2, 20),
            EngineChoice::Incremental
        );
        // Independent scalar oracle on the slow group's sample:
        // 95 * 100 >= 100 * (100 - 20), so its saving misses the margin.
        let slow = data.last().expect("slow group sample").observation.pair;
        assert!(u128::from(slow.incremental_refresh_ns) * 100 >= u128::from(slow.full_ns) * 80);
        assert_eq!(
            choose_with_group_balanced_margin(&data, target, 2, 2, 20),
            EngineChoice::Full
        );
    }

    #[test]
    fn independent_groups_with_strict_margin_are_admitted() {
        let data = [row(1, 1, 12, 20), row(2, 2, 20, 30)];
        assert_eq!(
            choose_with_group_balanced_margin(
                &data,
                Geometry {
                    gates: 100,
                    cone: 10
                },
                2,
                2,
                20
            ),
            EngineChoice::Incremental
        );
    }

    #[test]
    fn equal_to_margin_is_rejected() {
        let data = [row(1, 1, 12, 20), row(2, 2, 20, 80)];
        assert_eq!(
            choose_with_group_balanced_margin(
                &data,
                Geometry {
                    gates: 100,
                    cone: 10
                },
                2,
                2,
                20
            ),
            EngineChoice::Full
        );
    }
}

/// One *sequential* accepted or rejected intervention. Costs include any
/// cache reconstruction or refresh performed after an accepted intervention.
/// The oracle must compare the complete resulting state, not only outputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SequentialCost {
    pub full_ns: u64,
    pub incremental_ns: u64,
    pub cache_update_ns: u64,
    pub accepted: bool,
    pub exact_state_equal: bool,
}

/// Admit incremental execution only if the entire VALID sequence, including
/// post-acceptance cache maintenance, beats Full by a strict integer margin.
/// Rejected interventions still count as attempted work. This is deliberately
/// independent of the per-geometry gate; callers must satisfy *both* gates.
/// No timings from TEST/OOD may be passed to this function.
#[must_use]
pub fn choose_for_sequential_costs(
    valid_sequence: &[SequentialCost],
    minimum_steps: usize,
    minimum_saving_percent: u8,
) -> EngineChoice {
    if minimum_steps == 0 || valid_sequence.len() < minimum_steps || minimum_saving_percent > 100 {
        return EngineChoice::Full;
    }
    let mut full = 0_u128;
    let mut incremental = 0_u128;
    for step in valid_sequence {
        if !step.exact_state_equal || step.full_ns == 0 {
            return EngineChoice::Full;
        }
        // Rejected edits must not trigger post-acceptance reconstruction.
        if !step.accepted && step.cache_update_ns != 0 {
            return EngineChoice::Full;
        }
        let Some(next_full) = full.checked_add(u128::from(step.full_ns)) else {
            return EngineChoice::Full;
        };
        full = next_full;
        let Some(step_cost) =
            u128::from(step.incremental_ns).checked_add(u128::from(step.cache_update_ns))
        else {
            return EngineChoice::Full;
        };
        let Some(next_incremental) = incremental.checked_add(step_cost) else {
            return EngineChoice::Full;
        };
        incremental = next_incremental;
        // Fail closed on a prefix regression: a late fast step cannot hide
        // earlier sequential stalls.
        if incremental >= full {
            return EngineChoice::Full;
        }
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
mod sequential_cost_tests {
    use super::*;

    fn step(full: u64, incremental: u64, refresh: u64, accepted: bool) -> SequentialCost {
        SequentialCost {
            full_ns: full,
            incremental_ns: incremental,
            cache_update_ns: refresh,
            accepted,
            exact_state_equal: true,
        }
    }

    #[test]
    fn accepted_refresh_is_charged_and_can_reverse_dispatch() {
        let valid = [step(100, 10, 95, true), step(100, 10, 0, false)];
        assert_eq!(
            choose_for_sequential_costs(&valid, 2, 20),
            EngineChoice::Full
        );
        let fast = [step(100, 10, 20, true), step(100, 10, 0, false)];
        assert_eq!(
            choose_for_sequential_costs(&fast, 2, 20),
            EngineChoice::Incremental
        );
    }

    #[test]
    fn prefix_regression_is_not_hidden_by_later_fast_steps() {
        let valid = [step(100, 101, 0, true), step(1000, 1, 0, false)];
        assert_eq!(
            choose_for_sequential_costs(&valid, 2, 20),
            EngineChoice::Full
        );
    }

    #[test]
    fn rejected_refresh_and_oracle_mismatch_fail_closed() {
        let bad_refresh = [step(100, 1, 1, false)];
        assert_eq!(
            choose_for_sequential_costs(&bad_refresh, 1, 0),
            EngineChoice::Full
        );
        let mut mismatch = [step(100, 1, 1, true)];
        mismatch[0].exact_state_equal = false;
        assert_eq!(
            choose_for_sequential_costs(&mismatch, 1, 0),
            EngineChoice::Full
        );
    }

    #[test]
    fn strict_boundary_and_invalid_configuration() {
        let valid = [step(100, 80, 0, true)];
        assert_eq!(
            choose_for_sequential_costs(&valid, 1, 20),
            EngineChoice::Full
        );
        assert_eq!(
            choose_for_sequential_costs(&valid, 1, 19),
            EngineChoice::Incremental
        );
        assert_eq!(
            choose_for_sequential_costs(&valid, 0, 19),
            EngineChoice::Full
        );
        assert_eq!(
            choose_for_sequential_costs(&valid, 1, 101),
            EngineChoice::Full
        );
    }
}

/// A measured VALID sequential workload, including the peak resident memory
/// observed for each engine. VmHWM values are in KiB, not bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SequentialMemorySample {
    pub cost: SequentialCost,
    pub full_vmhwm_kib: u64,
    pub incremental_vmhwm_kib: u64,
}

/// Conservative dispatch for heterogeneous sequential workloads: every
/// prefix must remain faster, exact-state equality is mandatory, and the
/// incremental peak must fit the declared memory ceiling. Peak RSS is
/// compared per paired run; it is not summed across sequential steps.
/// Memory figures must come from isolated, identically bounded runs.
#[must_use]
pub fn choose_for_sequential_memory(
    valid: &[SequentialMemorySample],
    minimum_steps: usize,
    minimum_saving_percent: u8,
    max_incremental_vmhwm_kib: u64,
) -> EngineChoice {
    if max_incremental_vmhwm_kib == 0 {
        return EngineChoice::Full;
    }
    let mut costs = Vec::with_capacity(valid.len());
    for sample in valid {
        if sample.full_vmhwm_kib == 0
            || sample.incremental_vmhwm_kib == 0
            || sample.incremental_vmhwm_kib > max_incremental_vmhwm_kib
        {
            return EngineChoice::Full;
        }
        costs.push(sample.cost);
    }
    choose_for_sequential_costs(&costs, minimum_steps, minimum_saving_percent)
}

#[cfg(test)]
mod sequential_memory_tests {
    use super::*;

    fn sample(full: u64, incremental: u64, refresh: u64, rss: u64) -> SequentialMemorySample {
        SequentialMemorySample {
            cost: SequentialCost {
                full_ns: full,
                incremental_ns: incremental,
                cache_update_ns: refresh,
                accepted: true,
                exact_state_equal: true,
            },
            full_vmhwm_kib: 100,
            incremental_vmhwm_kib: rss,
        }
    }

    #[test]
    fn heterogeneous_sequence_admitted_with_exact_refresh_cost_and_memory() {
        let data = [sample(100, 10, 20, 120), sample(500, 100, 30, 140)];
        assert_eq!(
            choose_for_sequential_memory(&data, 2, 20, 150),
            EngineChoice::Incremental
        );
    }

    #[test]
    fn peak_memory_exceeding_limit_forces_full_despite_speedup() {
        let data = [sample(100, 1, 1, 151), sample(500, 1, 1, 140)];
        assert_eq!(
            choose_for_sequential_memory(&data, 2, 20, 150),
            EngineChoice::Full
        );
    }

    #[test]
    fn missing_memory_measurement_fails_closed() {
        let data = [sample(100, 1, 1, 0)];
        assert_eq!(
            choose_for_sequential_memory(&data, 1, 20, 150),
            EngineChoice::Full
        );
    }

    #[test]
    fn exact_state_oracle_mismatch_overrides_memory_and_speed() {
        let mut data = [sample(100, 1, 1, 120)];
        data[0].cost.exact_state_equal = false;
        assert_eq!(
            choose_for_sequential_memory(&data, 1, 20, 150),
            EngineChoice::Full
        );
    }
}

/// A sequential measurement is bound to the stable VALID sample identity.
/// Positional joins are unsafe because shuffled or cross-split timings can
/// otherwise silently calibrate the dispatch policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IdentifiedSequentialMemorySample {
    pub sample_id: u64,
    pub measurement: SequentialMemorySample,
}

/// Joint admission requires independent VALID groups and exactly one measured
/// sequential cost for each VALID sample, in intervention order. No TRAIN,
/// TEST or OOD measurement can enter the sequential policy. This is an
/// admission contract, not a benchmark or a cache implementation.
#[must_use]
pub fn choose_with_bound_sequential_valid(
    observations: &[GroupedSplitPair],
    valid_sequence: &[IdentifiedSequentialMemorySample],
    target: Geometry,
    minimum_distinct: usize,
    minimum_samples: usize,
    minimum_saving_percent: u8,
    max_incremental_vmhwm_kib: u64,
) -> EngineChoice {
    if choose_with_group_balanced_margin(
        observations,
        target,
        minimum_distinct,
        minimum_samples,
        minimum_saving_percent,
    ) != EngineChoice::Incremental
    {
        return EngineChoice::Full;
    }
    let valid_count = observations
        .iter()
        .filter(|row| row.observation.split == Split::Valid)
        .count();
    if valid_count != valid_sequence.len() {
        return EngineChoice::Full;
    }
    // The VALID observations encode intervention order. An arbitrary
    // permutation changes cache history, even when the ID sets match.
    // Check the ordered identity stream before evaluating sequential costs.
    let mut valid_ids = observations
        .iter()
        .filter(|row| row.observation.split == Split::Valid)
        .map(|row| row.observation.sample_id);
    let mut measurements = Vec::with_capacity(valid_count);
    for item in valid_sequence {
        if valid_ids.next() != Some(item.sample_id) {
            return EngineChoice::Full;
        }
        measurements.push(item.measurement);
    }
    choose_for_sequential_memory(
        &measurements,
        minimum_samples,
        minimum_saving_percent,
        max_incremental_vmhwm_kib,
    )
}

#[cfg(test)]
mod bound_sequential_tests {
    use super::*;

    fn group(id: u64, group_id: u64, split: Split, cone: u64) -> GroupedSplitPair {
        GroupedSplitPair {
            group_id,
            observation: SplitPair {
                sample_id: id,
                split,
                pair: ValidPair {
                    geometry: Geometry { gates: 100, cone },
                    full_ns: 100,
                    incremental_refresh_ns: 10,
                    oracle_equal: true,
                },
            },
        }
    }

    fn measurement(id: u64) -> IdentifiedSequentialMemorySample {
        IdentifiedSequentialMemorySample {
            sample_id: id,
            measurement: SequentialMemorySample {
                cost: SequentialCost {
                    full_ns: 100,
                    incremental_ns: 10,
                    cache_update_ns: 10,
                    accepted: true,
                    exact_state_equal: true,
                },
                full_vmhwm_kib: 100,
                incremental_vmhwm_kib: 120,
            },
        }
    }

    #[test]
    fn valid_grouped_sequence_is_admitted() {
        let rows = [
            group(1, 10, Split::Valid, 12),
            group(2, 20, Split::Valid, 20),
            group(3, 30, Split::Test, 12),
        ];
        assert_eq!(
            choose_with_bound_sequential_valid(
                &rows,
                &[measurement(1), measurement(2)],
                Geometry {
                    gates: 100,
                    cone: 10
                },
                2,
                2,
                20,
                150,
            ),
            EngineChoice::Incremental
        );
    }

    #[test]
    fn test_timing_and_duplicate_identity_are_rejected() {
        let rows = [
            group(1, 10, Split::Valid, 12),
            group(2, 20, Split::Valid, 20),
            group(3, 30, Split::Test, 12),
        ];
        let target = Geometry {
            gates: 100,
            cone: 10,
        };
        for timings in [
            [measurement(1), measurement(3)],
            [measurement(1), measurement(1)],
        ] {
            assert_eq!(
                choose_with_bound_sequential_valid(&rows, &timings, target, 2, 2, 20, 150),
                EngineChoice::Full
            );
        }
    }

    #[test]
    fn permuted_valid_interventions_are_not_a_valid_cache_history() {
        let rows = [
            group(1, 10, Split::Valid, 12),
            group(2, 20, Split::Valid, 20),
        ];
        let target = Geometry {
            gates: 100,
            cone: 10,
        };
        // Independent order oracle: the two identity streams differ.
        let expected_order = [1_u64, 2];
        let actual_order = [2_u64, 1];
        assert_ne!(expected_order, actual_order);
        assert_eq!(
            choose_with_bound_sequential_valid(
                &rows,
                &[measurement(2), measurement(1)],
                target,
                2,
                2,
                20,
                150,
            ),
            EngineChoice::Full
        );
        assert_eq!(
            choose_with_bound_sequential_valid(
                &rows,
                &[measurement(1), measurement(2)],
                target,
                2,
                2,
                20,
                150,
            ),
            EngineChoice::Incremental
        );
    }

    #[test]
    fn incomplete_sequence_and_oracle_divergence_are_rejected() {
        let rows = [
            group(1, 10, Split::Valid, 12),
            group(2, 20, Split::Valid, 20),
        ];
        let target = Geometry {
            gates: 100,
            cone: 10,
        };
        assert_eq!(
            choose_with_bound_sequential_valid(&rows, &[measurement(1)], target, 2, 2, 20, 150),
            EngineChoice::Full
        );
        let mut bad = [measurement(1), measurement(2)];
        bad[1].measurement.cost.exact_state_equal = false;
        assert_eq!(
            choose_with_bound_sequential_valid(&rows, &bad, target, 2, 2, 20, 150),
            EngineChoice::Full
        );
    }
}
