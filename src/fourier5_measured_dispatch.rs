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
