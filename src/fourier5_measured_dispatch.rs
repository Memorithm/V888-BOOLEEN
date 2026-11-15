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
