//! Conservative measured Full/Incremental dispatch for synthetic sequential circuits.
//! Calibration data must come from TRAIN; VALID/TEST/OOD are read-only evaluations.
//! No biological interpretation or automatic training on held-out samples.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineChoice { Full, Incremental }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConeBucket {
    pub gates: usize,
    pub cone_gates: usize,
    pub accepted: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PairedCost {
    pub bucket: ConeBucket,
    pub full_ns: u128,
    pub incremental_ns: u128,
    pub same_output: bool,
    pub same_decision: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PolicyConfig {
    pub min_pairs: usize,
    /// Incremental must save at least this fraction of Full cost (basis points).
    pub required_saving_bps: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyError { InvalidConfiguration, InvalidSample, Overflow }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BucketPolicy {
    pub bucket: ConeBucket,
    pub choice: EngineChoice,
    pub pairs: usize,
    pub full_total_ns: u128,
    pub incremental_total_ns: u128,
}

/// Fail closed: any oracle disagreement, zero-duration sample, insufficient
/// calibration or missing bucket keeps the independent Full engine.
pub fn calibrate(
    bucket: ConeBucket,
    train: &[PairedCost],
    config: PolicyConfig,
) -> Result<BucketPolicy, PolicyError> {
    if config.min_pairs == 0 || config.required_saving_bps > 10_000
        || bucket.gates == 0 || bucket.cone_gates > bucket.gates {
        return Err(PolicyError::InvalidConfiguration);
    }
    let mut pairs = 0usize;
    let mut full_total_ns = 0u128;
    let mut incremental_total_ns = 0u128;
    let mut valid = true;
    for sample in train.iter().filter(|sample| sample.bucket == bucket) {
        pairs = pairs.checked_add(1).ok_or(PolicyError::Overflow)?;
        if sample.full_ns == 0 || sample.incremental_ns == 0 {
            return Err(PolicyError::InvalidSample);
        }
        full_total_ns = full_total_ns.checked_add(sample.full_ns).ok_or(PolicyError::Overflow)?;
        incremental_total_ns = incremental_total_ns.checked_add(sample.incremental_ns)
            .ok_or(PolicyError::Overflow)?;
        valid &= sample.same_output && sample.same_decision;
    }
    // Division avoids multiplication overflow: incremental <= full * (1 - margin).
    // Use a checked product for exact integer comparison; overflow fails closed.
    let threshold = u128::from(10_000 - config.required_saving_bps);
    let faster = match (
        incremental_total_ns.checked_mul(10_000),
        full_total_ns.checked_mul(threshold),
    ) {
        (Some(inc), Some(full)) => inc <= full,
        _ => false,
    };
    let choice = if valid && pairs >= config.min_pairs && faster {
        EngineChoice::Incremental
    } else {
        EngineChoice::Full
    };
    Ok(BucketPolicy { bucket, choice, pairs, full_total_ns, incremental_total_ns })
}

pub fn select(bucket: ConeBucket, policies: &[BucketPolicy]) -> EngineChoice {
    // Ambiguous duplicate calibration must never choose Incremental.
    let mut matching = policies.iter().filter(|policy| policy.bucket == bucket);
    match (matching.next(), matching.next()) {
        (Some(policy), None) => policy.choice,
        _ => EngineChoice::Full,
    }
}

/// Read-only holdout audit: it does not mutate calibration or choose thresholds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HoldoutAudit {
    pub samples: usize,
    pub mismatches: usize,
    pub incremental_chosen: usize,
    pub regressions: usize,
}
pub fn audit_holdout(samples: &[PairedCost], policies: &[BucketPolicy])
    -> Result<HoldoutAudit, PolicyError>
{
    let mut audit = HoldoutAudit {
        samples: 0, mismatches: 0, incremental_chosen: 0, regressions: 0,
    };
    for sample in samples {
        audit.samples = audit.samples.checked_add(1).ok_or(PolicyError::Overflow)?;
        if !sample.same_output || !sample.same_decision {
            audit.mismatches = audit.mismatches.checked_add(1).ok_or(PolicyError::Overflow)?;
        }
        if select(sample.bucket, policies) == EngineChoice::Incremental {
            audit.incremental_chosen = audit.incremental_chosen.checked_add(1)
                .ok_or(PolicyError::Overflow)?;
            if sample.incremental_ns > sample.full_ns {
                audit.regressions = audit.regressions.checked_add(1)
                    .ok_or(PolicyError::Overflow)?;
            }
        }
    }
    Ok(audit)
}

#[cfg(test)]
mod tests {
    use super::*;
    const BUCKET: ConeBucket = ConeBucket { gates: 100, cone_gates: 4, accepted: true };
    fn sample(full: u128, inc: u128, same: bool) -> PairedCost {
        PairedCost { bucket: BUCKET, full_ns: full, incremental_ns: inc,
            same_output: same, same_decision: same }
    }
    const CONFIG: PolicyConfig = PolicyConfig { min_pairs: 2, required_saving_bps: 2000 };
    #[test]
    fn measured_dispatch_and_independent_arithmetic_oracle() {
        let train = [sample(100, 30, true), sample(200, 50, true)];
        let policy = calibrate(BUCKET, &train, CONFIG).unwrap();
        assert_eq!(policy.choice, EngineChoice::Incremental);
        assert_eq!(policy.full_total_ns, 300);
        assert_eq!(policy.incremental_total_ns, 80);
        assert_eq!(select(BUCKET, &[policy]), EngineChoice::Incremental);
        assert_eq!(select(BUCKET, &[policy, policy]), EngineChoice::Full);
        let heldout = [sample(10, 20, true), sample(100, 10, false)];
        let report = audit_holdout(&heldout, &[policy]).unwrap();
        assert_eq!(report.samples, 2);
        assert_eq!(report.mismatches, 1);
        assert_eq!(report.regressions, 1);
        assert_eq!(report.incremental_chosen, 2);
        // No calibration mutation after holdout evaluation.
        assert_eq!(policy.full_total_ns, 300);
    }
    #[test]
    fn fail_closed_on_mismatch_small_sample_and_overflow() {
        assert_eq!(calibrate(BUCKET, &[sample(100, 1, true)], CONFIG).unwrap().choice,
            EngineChoice::Full);
        assert_eq!(calibrate(BUCKET, &[sample(100, 1, false), sample(100, 1, true)], CONFIG)
            .unwrap().choice, EngineChoice::Full);
        assert_eq!(calibrate(BUCKET, &[sample(u128::MAX, 1, true),
            sample(1, 1, true)], CONFIG), Err(PolicyError::Overflow));
        assert_eq!(calibrate(BUCKET, &[sample(0, 1, true)], CONFIG),
            Err(PolicyError::InvalidSample));
        assert_eq!(select(ConeBucket { accepted: false, ..BUCKET }, &[]),
            EngineChoice::Full);
    }
}


/// Freeze a TRAIN-calibrated policy against a disjoint VALID slice.
/// The gate never tunes thresholds or learns from validation timings.
/// Every VALID observation for this bucket must preserve outputs/decisions,
/// and an Incremental choice must not regress on any individual sample.
/// Missing VALID coverage fails closed to Full.
pub fn gate_on_validation(
    trained: BucketPolicy,
    validation: &[PairedCost],
    min_validation_pairs: usize,
) -> Result<BucketPolicy, PolicyError> {
    if min_validation_pairs == 0 {
        return Err(PolicyError::InvalidConfiguration);
    }
    let mut observed = 0usize;
    let mut safe = true;
    for sample in validation.iter().filter(|sample| sample.bucket == trained.bucket) {
        observed = observed.checked_add(1).ok_or(PolicyError::Overflow)?;
        if sample.full_ns == 0 || sample.incremental_ns == 0 {
            return Err(PolicyError::InvalidSample);
        }
        safe &= sample.same_output
            && sample.same_decision
            && sample.incremental_ns <= sample.full_ns;
    }
    let choice = if trained.choice == EngineChoice::Incremental
        && observed >= min_validation_pairs && safe {
        EngineChoice::Incremental
    } else {
        EngineChoice::Full
    };
    Ok(BucketPolicy { choice, ..trained })
}

#[cfg(test)]
mod validation_gate_tests {
    use super::*;

    const B: ConeBucket = ConeBucket { gates: 32, cone_gates: 3, accepted: true };
    fn paired(full: u128, inc: u128, same: bool) -> PairedCost {
        PairedCost { bucket: B, full_ns: full, incremental_ns: inc,
            same_output: same, same_decision: same }
    }

    #[test]
    fn disjoint_validation_never_recalibrates_and_rejects_single_regression() {
        let train = [paired(100, 20, true), paired(200, 40, true)];
        let frozen = calibrate(B, &train, PolicyConfig {
            min_pairs: 2, required_saving_bps: 1000,
        }).unwrap();
        assert_eq!(frozen.choice, EngineChoice::Incremental);
        let valid = [paired(100, 10, true), paired(50, 49, true)];
        assert_eq!(gate_on_validation(frozen, &valid, 2).unwrap(), frozen);
        // The aggregate still improves, but one slower accepted update fails.
        let regressed = [paired(100, 10, true), paired(50, 51, true)];
        let guarded = gate_on_validation(frozen, &regressed, 2).unwrap();
        assert_eq!(guarded.choice, EngineChoice::Full);
        assert_eq!(guarded.full_total_ns, frozen.full_total_ns);
        assert_eq!(guarded.incremental_total_ns, frozen.incremental_total_ns);
        assert_eq!(gate_on_validation(frozen, &valid[..1], 2).unwrap().choice,
            EngineChoice::Full);
        assert_eq!(gate_on_validation(frozen, &[paired(100, 10, false)], 1)
            .unwrap().choice, EngineChoice::Full);
        assert_eq!(gate_on_validation(frozen, &[], 0),
            Err(PolicyError::InvalidConfiguration));
    }
}


/// Freeze a whole dispatch table after VALID, rejecting duplicate buckets.
/// Every trained bucket is independently gated; a missing VALID bucket
/// conservatively resolves to Full. TEST/OOD must never be passed here.
pub fn freeze_validated_table(
    trained: &[BucketPolicy],
    validation: &[PairedCost],
    min_validation_pairs: usize,
) -> Result<Vec<BucketPolicy>, PolicyError> {
    if min_validation_pairs == 0 {
        return Err(PolicyError::InvalidConfiguration);
    }
    // Duplicate buckets make dispatch ambiguous, even if choices agree.
    for (index, policy) in trained.iter().enumerate() {
        if trained[..index].iter().any(|prior| prior.bucket == policy.bucket) {
            return Err(PolicyError::InvalidConfiguration);
        }
        if policy.bucket.gates == 0 || policy.bucket.cone_gates > policy.bucket.gates {
            return Err(PolicyError::InvalidConfiguration);
        }
    }
    // A malformed or uncalibrated VALID observation must not be silently
    // ignored just because its bucket was absent from TRAIN.
    for sample in validation {
        if sample.full_ns == 0 || sample.incremental_ns == 0 {
            return Err(PolicyError::InvalidSample);
        }
    }
    trained.iter().copied().map(|policy| {
        gate_on_validation(policy, validation, min_validation_pairs)
    }).collect()
}

/// The frozen table is the only source of dispatch choices. An unseen or
/// duplicated bucket is always Full; no held-out observation changes it.
pub fn select_frozen(bucket: ConeBucket, frozen: &[BucketPolicy]) -> EngineChoice {
    select(bucket, frozen)
}

#[cfg(test)]
mod frozen_table_tests {
    use super::*;

    #[test]
    fn disjoint_bucket_coverage_and_no_holdout_recalibration() {
        let a = ConeBucket { gates: 128, cone_gates: 2, accepted: true };
        let b = ConeBucket { gates: 128, cone_gates: 40, accepted: false };
        let config = PolicyConfig { min_pairs: 2, required_saving_bps: 1000 };
        let make = |bucket, full_ns, incremental_ns| PairedCost {
            bucket, full_ns, incremental_ns, same_output: true, same_decision: true,
        };
        let train = [
            make(a, 100, 20), make(a, 100, 20),
            make(b, 100, 20), make(b, 100, 20),
        ];
        let trained = [
            calibrate(a, &train, config).unwrap(),
            calibrate(b, &train, config).unwrap(),
        ];
        // VALID is disjoint from TRAIN by caller contract. Bucket b has no
        // coverage, and must not inherit bucket a's successful validation.
        let valid = [make(a, 90, 30), make(a, 110, 40)];
        let frozen = freeze_validated_table(&trained, &valid, 2).unwrap();
        assert_eq!(select_frozen(a, &frozen), EngineChoice::Incremental);
        assert_eq!(select_frozen(b, &frozen), EngineChoice::Full);
        assert_eq!(select_frozen(ConeBucket { cone_gates: 3, ..a }, &frozen),
            EngineChoice::Full);
        assert_eq!(frozen[0].full_total_ns, trained[0].full_total_ns);
        assert_eq!(frozen[0].incremental_total_ns, trained[0].incremental_total_ns);
        assert_eq!(freeze_validated_table(&[trained[0], trained[0]], &valid, 2),
            Err(PolicyError::InvalidConfiguration));
        assert_eq!(freeze_validated_table(&trained, &[make(b, 0, 1)], 1),
            Err(PolicyError::InvalidSample));
        assert_eq!(freeze_validated_table(&trained, &valid, 0),
            Err(PolicyError::InvalidConfiguration));
    }

    #[test]
    fn single_validation_regression_blocks_only_its_bucket() {
        let a = ConeBucket { gates: 64, cone_gates: 1, accepted: true };
        let b = ConeBucket { gates: 64, cone_gates: 2, accepted: false };
        let trained = [a, b].map(|bucket| BucketPolicy {
            bucket, choice: EngineChoice::Incremental, pairs: 4,
            full_total_ns: 400, incremental_total_ns: 100,
        });
        let valid = [a, b].map(|bucket| PairedCost {
            bucket, full_ns: 100,
            incremental_ns: if bucket == a { 101 } else { 50 },
            same_output: true, same_decision: true,
        });
        let frozen = freeze_validated_table(&trained, &valid, 1).unwrap();
        assert_eq!(select_frozen(a, &frozen), EngineChoice::Full);
        assert_eq!(select_frozen(b, &frozen), EngineChoice::Incremental);
    }
}


/// Choose before an intervention's acceptance is known. Acceptance is an
/// outcome of execution, so dispatch must not use its future value.
/// Incremental is permitted only when BOTH validated outcome buckets agree.
/// This is conservative for heterogeneous sequential workloads.
pub fn select_before_decision(
    gates: usize,
    cone_gates: usize,
    frozen: &[BucketPolicy],
) -> EngineChoice {
    if gates == 0 || cone_gates > gates {
        return EngineChoice::Full;
    }
    let accepted = ConeBucket { gates, cone_gates, accepted: true };
    let rejected = ConeBucket { gates, cone_gates, accepted: false };
    if select_frozen(accepted, frozen) == EngineChoice::Incremental
        && select_frozen(rejected, frozen) == EngineChoice::Incremental
    {
        EngineChoice::Incremental
    } else {
        EngineChoice::Full
    }
}

#[cfg(test)]
mod predecision_dispatch_tests {
    use super::*;

    #[test]
    fn unknown_acceptance_cannot_leak_into_dispatch() {
        let yes = ConeBucket { gates: 100, cone_gates: 3, accepted: true };
        let no = ConeBucket { accepted: false, ..yes };
        let policy = |bucket, choice| BucketPolicy {
            bucket, choice, pairs: 3, full_total_ns: 300,
            incremental_total_ns: 60,
        };
        let both = [policy(yes, EngineChoice::Incremental),
                    policy(no, EngineChoice::Incremental)];
        assert_eq!(select_before_decision(100, 3, &both), EngineChoice::Incremental);
        assert_eq!(select_before_decision(100, 3, &both[..1]), EngineChoice::Full);
        let mixed = [both[0], policy(no, EngineChoice::Full)];
        assert_eq!(select_before_decision(100, 3, &mixed), EngineChoice::Full);
        assert_eq!(select_before_decision(100, 4, &both), EngineChoice::Full);
        assert_eq!(select_before_decision(0, 0, &both), EngineChoice::Full);
        assert_eq!(select_before_decision(100, 101, &both), EngineChoice::Full);
        assert_eq!(select_before_decision(100, 3, &[both[0], both[0], both[1]]),
                   EngineChoice::Full);
    }
}


/// Provenance identifiers for independently generated circuit/workload cases.
/// IDs must be stable across engines and all four splits. Reject repeated IDs
/// even within one split: repeated interventions are not independent samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WorkloadId {
    pub circuit_seed: u64,
    pub workload_seed: u64,
}

/// A measured observation paired with its immutable experimental identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IdentifiedCost {
    pub id: WorkloadId,
    pub cost: PairedCost,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitError {
    DuplicateOrLeakedWorkload,
    LeakedCircuitAcrossSplits,
    InvalidObservation,
    Policy(PolicyError),
}

/// Calibrate only on TRAIN, gate only on VALID, and audit TEST/OOD without
/// changing the frozen table. No split can silently reuse a workload ID.
/// This is a provenance guard, not a substitute for independent data generation.
pub fn evaluate_disjoint_splits(
    train: &[IdentifiedCost],
    valid: &[IdentifiedCost],
    test: &[IdentifiedCost],
    ood: &[IdentifiedCost],
    buckets: &[ConeBucket],
    config: PolicyConfig,
    min_valid_pairs: usize,
) -> Result<(Vec<BucketPolicy>, HoldoutAudit, HoldoutAudit), SplitError> {
    let mut ids = std::collections::HashSet::new();
    // A different intervention seed on the same synthetic circuit is not an
    // independent holdout: the circuit topology itself can be memorised.
    // Permit multiple interventions on one circuit within a split, but never
    // allow that circuit to cross the TRAIN/VALID/TEST/OOD boundary.
    let mut circuit_owner = std::collections::HashMap::<u64, usize>::new();
    for (split_index, split) in [train, valid, test, ood].into_iter().enumerate() {
        for observation in split {
            match circuit_owner.entry(observation.id.circuit_seed) {
                std::collections::hash_map::Entry::Vacant(entry) => { entry.insert(split_index); }
                std::collections::hash_map::Entry::Occupied(entry)
                    if *entry.get() != split_index => {
                        return Err(SplitError::LeakedCircuitAcrossSplits);
                    }
                _ => {}
            }
            if !ids.insert(observation.id) {
                return Err(SplitError::DuplicateOrLeakedWorkload);
            }
            let cost = observation.cost;
            if cost.bucket.gates == 0
                || cost.bucket.cone_gates > cost.bucket.gates
                || cost.full_ns == 0 || cost.incremental_ns == 0
            {
                return Err(SplitError::InvalidObservation);
            }
        }
    }
    if buckets.iter().enumerate().any(|(i, bucket)| {
        buckets[..i].contains(bucket)
    }) {
        return Err(SplitError::Policy(PolicyError::InvalidConfiguration));
    }
    let train_costs: Vec<_> = train.iter().map(|item| item.cost).collect();
    let valid_costs: Vec<_> = valid.iter().map(|item| item.cost).collect();
    let trained: Vec<_> = buckets.iter().map(|&bucket| {
        calibrate(bucket, &train_costs, config).map_err(SplitError::Policy)
    }).collect::<Result<_, _>>()?;
    let frozen = freeze_validated_table(&trained, &valid_costs, min_valid_pairs)
        .map_err(SplitError::Policy)?;
    let test_costs: Vec<_> = test.iter().map(|item| item.cost).collect();
    let ood_costs: Vec<_> = ood.iter().map(|item| item.cost).collect();
    let test_audit = audit_predecision_holdout(&test_costs, &frozen).map_err(SplitError::Policy)?;
    let ood_audit = audit_predecision_holdout(&ood_costs, &frozen).map_err(SplitError::Policy)?;
    Ok((frozen, test_audit, ood_audit))
}

#[cfg(test)]
mod disjoint_split_tests {
    use super::*;

    #[test]
    fn split_guard_rejects_cross_split_and_internal_reuse() {
        let bucket = ConeBucket { gates: 40, cone_gates: 2, accepted: true };
        let make = |seed, full_ns, incremental_ns| IdentifiedCost {
            id: WorkloadId { circuit_seed: seed, workload_seed: 7 },
            cost: PairedCost { bucket, full_ns, incremental_ns,
                same_output: true, same_decision: true },
        };
        let train = [make(1, 100, 20), make(2, 100, 20)];
        let valid = [make(3, 100, 30)];
        let test = [make(4, 100, 110)];
        let ood = [make(5, 100, 40)];
        let config = PolicyConfig { min_pairs: 2, required_saving_bps: 1000 };
        let (frozen, test_audit, ood_audit) = evaluate_disjoint_splits(
            &train, &valid, &test, &ood, &[bucket], config, 1
        ).unwrap();
        assert_eq!(select_frozen(bucket, &frozen), EngineChoice::Incremental);
        assert_eq!(test_audit.regressions, 0);
        assert_eq!(test_audit.incremental_chosen, 0);
        assert_eq!(ood_audit.regressions, 0);
        assert_eq!(evaluate_disjoint_splits(
            &train, &valid, &train[..1], &ood, &[bucket], config, 1
        ), Err(SplitError::LeakedCircuitAcrossSplits));
        assert_eq!(evaluate_disjoint_splits(
            &train, &[valid[0], valid[0]], &test, &ood, &[bucket], config, 1
        ), Err(SplitError::DuplicateOrLeakedWorkload));
        assert_eq!(evaluate_disjoint_splits(
            &train, &valid, &test, &ood, &[bucket, bucket], config, 1
        ), Err(SplitError::Policy(PolicyError::InvalidConfiguration)));
    }
}


/// Holdout audit using only pre-intervention information for dispatch.
/// The accepted flag is an observed outcome, never an input to selection.
/// This avoids optimistic evaluation from hindsight-selected buckets.
pub fn audit_predecision_holdout(
    samples: &[PairedCost],
    frozen: &[BucketPolicy],
) -> Result<HoldoutAudit, PolicyError> {
    let mut audit = HoldoutAudit {
        samples: 0, mismatches: 0, incremental_chosen: 0, regressions: 0,
    };
    for sample in samples {
        if sample.bucket.gates == 0
            || sample.bucket.cone_gates > sample.bucket.gates
            || sample.full_ns == 0 || sample.incremental_ns == 0
        {
            return Err(PolicyError::InvalidSample);
        }
        audit.samples = audit.samples.checked_add(1).ok_or(PolicyError::Overflow)?;
        if !sample.same_output || !sample.same_decision {
            audit.mismatches = audit.mismatches.checked_add(1).ok_or(PolicyError::Overflow)?;
        }
        if select_before_decision(sample.bucket.gates, sample.bucket.cone_gates, frozen)
            == EngineChoice::Incremental
        {
            audit.incremental_chosen = audit.incremental_chosen
                .checked_add(1).ok_or(PolicyError::Overflow)?;
            if sample.incremental_ns > sample.full_ns {
                audit.regressions = audit.regressions
                    .checked_add(1).ok_or(PolicyError::Overflow)?;
            }
        }
    }
    Ok(audit)
}

#[cfg(test)]
mod predecision_holdout_tests {
    use super::*;

    #[test]
    fn oracle_rejects_hindsight_selection_and_reports_regressions() {
        let yes = ConeBucket { gates: 64, cone_gates: 3, accepted: true };
        let no = ConeBucket { accepted: false, ..yes };
        let policy = |bucket| BucketPolicy {
            bucket, choice: EngineChoice::Incremental, pairs: 2,
            full_total_ns: 200, incremental_total_ns: 80,
        };
        let observation = |bucket, full_ns, incremental_ns| PairedCost {
            bucket, full_ns, incremental_ns,
            same_output: true, same_decision: true,
        };
        let samples = [
            observation(yes, 100, 30),
            observation(no, 100, 120),
        ];
        // Outcome-aware audit would incorrectly count the accepted sample.
        let only_yes = [policy(yes)];
        assert_eq!(audit_holdout(&samples, &only_yes).unwrap().incremental_chosen, 1);
        assert_eq!(audit_predecision_holdout(&samples, &only_yes).unwrap().incremental_chosen, 0);
        let both = [policy(yes), policy(no)];
        let audited = audit_predecision_holdout(&samples, &both).unwrap();
        assert_eq!(audited.samples, 2);
        assert_eq!(audited.incremental_chosen, 2);
        assert_eq!(audited.regressions, 1);
        assert_eq!(audited.mismatches, 0);
        assert_eq!(audit_predecision_holdout(
            &[observation(no, 0, 20)], &both,
        ), Err(PolicyError::InvalidSample));
    }
}

#[cfg(test)]
mod circuit_isolation_tests {
    use super::*;

    #[test]
    fn same_circuit_different_intervention_cannot_cross_holdout_boundary() {
        let bucket = ConeBucket { gates: 12, cone_gates: 2, accepted: true };
        let make = |circuit_seed, workload_seed| IdentifiedCost {
            id: WorkloadId { circuit_seed, workload_seed },
            cost: PairedCost { bucket, full_ns: 100, incremental_ns: 20,
                same_output: true, same_decision: true },
        };
        let config = PolicyConfig { min_pairs: 1, required_saving_bps: 1000 };
        let train = [make(1, 10), make(1, 11)];
        let valid = [make(2, 10)];
        let test = [make(3, 10)];
        let ood = [make(4, 10)];
        assert!(evaluate_disjoint_splits(&train, &valid, &test, &ood,
            &[bucket], config, 1).is_ok());
        let leaked = [make(1, 99)];
        assert_eq!(evaluate_disjoint_splits(&train, &valid, &leaked, &ood,
            &[bucket], config, 1), Err(SplitError::LeakedCircuitAcrossSplits));
        // Exact duplicate inside TRAIN is still a separate provenance error.
        assert_eq!(evaluate_disjoint_splits(&[train[0], train[0]], &valid,
            &test, &ood, &[bucket], config, 1),
            Err(SplitError::DuplicateOrLeakedWorkload));
    }
}


/// Runtime cache-coherence gate for consecutive accepted interventions.
/// A cache is usable only when it describes the current committed circuit
/// revision. Rejected interventions do not advance that revision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheRevision {
    pub circuit_revision: u64,
    pub cached_revision: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RevisionError {
    Overflow,
    FutureCache,
}

impl CacheRevision {
    /// Do not use a cache from a future revision, even if a policy is calibrated.
    pub fn choose(
        &self,
        gates: usize,
        cone_gates: usize,
        frozen: &[BucketPolicy],
    ) -> Result<EngineChoice, RevisionError> {
        if self.cached_revision.is_some_and(|revision| revision > self.circuit_revision) {
            return Err(RevisionError::FutureCache);
        }
        if self.cached_revision != Some(self.circuit_revision) {
            return Ok(EngineChoice::Full);
        }
        Ok(select_before_decision(gates, cone_gates, frozen))
    }

    /// Call only after a proposed intervention has been committed.
    /// The old cache is invalidated even if the revision counter overflows.
    pub fn committed(&mut self) -> Result<(), RevisionError> {
        self.cached_revision = None;
        self.circuit_revision = self.circuit_revision
            .checked_add(1)
            .ok_or(RevisionError::Overflow)?;
        Ok(())
    }

    /// A rejected intervention leaves the committed circuit unchanged.
    pub fn rejected(&mut self) {}

    /// Mark a cache rebuilt from the exact committed circuit revision.
    /// Callers must finish rebuilding before invoking this method.
    pub fn rebuilt(&mut self, revision: u64) -> Result<(), RevisionError> {
        if revision != self.circuit_revision {
            return Err(RevisionError::FutureCache);
        }
        self.cached_revision = Some(revision);
        Ok(())
    }
}

#[cfg(test)]
mod sequential_cache_revision_tests {
    use super::*;

    #[test]
    fn accepted_update_invalidates_cache_until_exact_rebuild() {
        let yes = ConeBucket { gates: 80, cone_gates: 5, accepted: true };
        let no = ConeBucket { accepted: false, ..yes };
        let policy = |bucket| BucketPolicy {
            bucket,
            choice: EngineChoice::Incremental,
            pairs: 3,
            full_total_ns: 300,
            incremental_total_ns: 60,
        };
        let frozen = [policy(yes), policy(no)];
        let mut state = CacheRevision {
            circuit_revision: 7,
            cached_revision: Some(7),
        };
        assert_eq!(state.choose(80, 5, &frozen), Ok(EngineChoice::Incremental));
        state.rejected();
        assert_eq!(state.choose(80, 5, &frozen), Ok(EngineChoice::Incremental));
        state.committed().unwrap();
        assert_eq!(state.circuit_revision, 8);
        assert_eq!(state.choose(80, 5, &frozen), Ok(EngineChoice::Full));
        assert_eq!(state.rebuilt(7), Err(RevisionError::FutureCache));
        assert_eq!(state.choose(80, 5, &frozen), Ok(EngineChoice::Full));
        state.rebuilt(8).unwrap();
        assert_eq!(state.choose(80, 5, &frozen), Ok(EngineChoice::Incremental));
        assert_eq!(state.choose(80, 6, &frozen), Ok(EngineChoice::Full));
    }

    #[test]
    fn stale_and_future_cache_fail_closed_and_overflow_invalidates() {
        let mut state = CacheRevision {
            circuit_revision: 10,
            cached_revision: Some(11),
        };
        assert_eq!(state.choose(1, 1, &[]), Err(RevisionError::FutureCache));
        state.cached_revision = Some(9);
        assert_eq!(state.choose(1, 1, &[]), Ok(EngineChoice::Full));
        state.circuit_revision = u64::MAX;
        state.cached_revision = Some(u64::MAX);
        assert_eq!(state.committed(), Err(RevisionError::Overflow));
        assert_eq!(state.cached_revision, None);
        assert_eq!(state.choose(1, 1, &[]), Ok(EngineChoice::Full));
    }
}


/// Rebuild a cache for the committed revision as a fail-closed transaction.
///
/// The callback must actually rebuild the caller-owned cache from the current
/// circuit. A failed callback never certifies the cache, even if it was valid
/// before the attempted rebuild. No synthetic timing is inferred here.
impl CacheRevision {
    pub fn rebuild_transaction<E>(
        &mut self,
        rebuild: impl FnOnce(u64) -> Result<(), E>,
    ) -> Result<(), E> {
        // Invalidate before executing arbitrary caller code. In particular,
        // a failed rebuild must not retain a previously valid certificate.
        self.cached_revision = None;
        let revision = self.circuit_revision;
        rebuild(revision)?;
        self.cached_revision = Some(revision);
        Ok(())
    }
}

#[cfg(test)]
mod transactional_cache_tests {
    use super::*;

    #[test]
    fn successful_rebuild_certifies_only_current_revision() {
        let mut state = CacheRevision {
            circuit_revision: 4,
            cached_revision: Some(4),
        };
        let mut observed = None;
        state.rebuild_transaction(|revision| {
            observed = Some(revision);
            Ok::<(), ()>(())
        }).unwrap();
        assert_eq!(observed, Some(4));
        assert_eq!(state.cached_revision, Some(4));
        state.committed().unwrap();
        assert_eq!(state.cached_revision, None);
        state.rebuild_transaction(|revision| {
            assert_eq!(revision, 5);
            Ok::<(), ()>(())
        }).unwrap();
        assert_eq!(state.cached_revision, Some(5));
    }

    #[test]
    fn failed_rebuild_revokes_even_previously_valid_cache() {
        let mut state = CacheRevision {
            circuit_revision: 9,
            cached_revision: Some(9),
        };
        let result = state.rebuild_transaction(|revision| {
            assert_eq!(revision, 9);
            Err::<(), _>("incomplete rebuild")
        });
        assert_eq!(result, Err("incomplete rebuild"));
        assert_eq!(state.cached_revision, None);
        assert_eq!(state.choose(100, 1, &[]), Ok(EngineChoice::Full));
    }
}


/// One real, paired sequential intervention. Timings must include cache
/// maintenance performed *after* a committed intervention, before the next
/// intervention is dispatched. VmHWM is read from the executing process,
/// not inferred from allocations or generated synthetically.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SequentialPairedObservation {
    pub accepted: bool,
    pub same_output: bool,
    pub same_decision: bool,
    pub full_execution_ns: u128,
    pub incremental_execution_ns: u128,
    pub full_refresh_ns: u128,
    pub incremental_refresh_ns: u128,
    pub full_vmhwm_kib: u64,
    pub incremental_vmhwm_kib: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SequentialPairedSummary {
    pub interventions: usize,
    pub accepted: usize,
    pub rejected: usize,
    pub full_total_ns: u128,
    pub incremental_total_ns: u128,
    pub full_peak_vmhwm_kib: u64,
    pub incremental_peak_vmhwm_kib: u64,
    pub mismatches: usize,
    pub regressions: usize,
}

/// Aggregate actual paired observations with checked arithmetic. Reject zero
/// execution durations and refresh work after rejected interventions; refresh
/// is charged to the same accepted intervention that invalidated the cache.
/// This deliberately does not choose a policy or manufacture a speedup.
pub fn summarize_sequential_pairs(
    samples: &[SequentialPairedObservation],
) -> Result<SequentialPairedSummary, PolicyError> {
    let mut out = SequentialPairedSummary::default();
    for sample in samples {
        if sample.full_execution_ns == 0 || sample.incremental_execution_ns == 0
            || (!sample.accepted
                && (sample.full_refresh_ns != 0 || sample.incremental_refresh_ns != 0))
        {
            return Err(PolicyError::InvalidSample);
        }
        let full = sample.full_execution_ns.checked_add(sample.full_refresh_ns)
            .ok_or(PolicyError::Overflow)?;
        let incremental = sample.incremental_execution_ns
            .checked_add(sample.incremental_refresh_ns)
            .ok_or(PolicyError::Overflow)?;
        out.interventions = out.interventions.checked_add(1).ok_or(PolicyError::Overflow)?;
        if sample.accepted {
            out.accepted = out.accepted.checked_add(1).ok_or(PolicyError::Overflow)?;
        } else {
            out.rejected = out.rejected.checked_add(1).ok_or(PolicyError::Overflow)?;
        }
        out.full_total_ns = out.full_total_ns.checked_add(full).ok_or(PolicyError::Overflow)?;
        out.incremental_total_ns = out.incremental_total_ns
            .checked_add(incremental).ok_or(PolicyError::Overflow)?;
        out.full_peak_vmhwm_kib = out.full_peak_vmhwm_kib.max(sample.full_vmhwm_kib);
        out.incremental_peak_vmhwm_kib =
            out.incremental_peak_vmhwm_kib.max(sample.incremental_vmhwm_kib);
        if !sample.same_output || !sample.same_decision {
            out.mismatches = out.mismatches.checked_add(1).ok_or(PolicyError::Overflow)?;
        }
        if incremental > full {
            out.regressions = out.regressions.checked_add(1).ok_or(PolicyError::Overflow)?;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod sequential_accounting_oracle_tests {
    use super::*;

    #[test]
    fn accepted_refresh_is_charged_and_negative_results_are_preserved() {
        let accepted = SequentialPairedObservation {
            accepted: true, same_output: true, same_decision: true,
            full_execution_ns: 100, incremental_execution_ns: 40,
            full_refresh_ns: 0, incremental_refresh_ns: 90,
            full_vmhwm_kib: 1000, incremental_vmhwm_kib: 1500,
        };
        let rejected = SequentialPairedObservation {
            accepted: false, same_output: true, same_decision: false,
            full_execution_ns: 80, incremental_execution_ns: 20,
            full_refresh_ns: 0, incremental_refresh_ns: 0,
            full_vmhwm_kib: 1100, incremental_vmhwm_kib: 1400,
        };
        let report = summarize_sequential_pairs(&[accepted, rejected]).unwrap();
        // Independent hand calculation: Full 100+80=180; Inc 40+90+20=150.
        assert_eq!((report.full_total_ns, report.incremental_total_ns), (180, 150));
        assert_eq!((report.accepted, report.rejected), (1, 1));
        assert_eq!((report.mismatches, report.regressions), (1, 1));
        assert_eq!((report.full_peak_vmhwm_kib, report.incremental_peak_vmhwm_kib),
            (1100, 1500));
        assert_eq!(summarize_sequential_pairs(&[]).unwrap().interventions, 0);
    }

    #[test]
    fn rejects_impossible_refresh_and_checked_overflow() {
        let base = SequentialPairedObservation {
            accepted: false, same_output: true, same_decision: true,
            full_execution_ns: 1, incremental_execution_ns: 1,
            full_refresh_ns: 0, incremental_refresh_ns: 0,
            full_vmhwm_kib: 1, incremental_vmhwm_kib: 1,
        };
        assert_eq!(summarize_sequential_pairs(&[
            SequentialPairedObservation { incremental_refresh_ns: 1, ..base }
        ]), Err(PolicyError::InvalidSample));
        assert_eq!(summarize_sequential_pairs(&[
            SequentialPairedObservation { accepted: true, full_execution_ns: u128::MAX,
                full_refresh_ns: 1, ..base }
        ]), Err(PolicyError::Overflow));
        assert_eq!(summarize_sequential_pairs(&[
            SequentialPairedObservation { full_execution_ns: u128::MAX, ..base },
            base,
        ]), Err(PolicyError::Overflow));
    }
}
