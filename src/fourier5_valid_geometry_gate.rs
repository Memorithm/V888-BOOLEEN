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
        // A pooled speedup can hide a slow cone. Reject any measured\n        // geometry whose incremental refresh is not strictly faster.\n        if pair.full_ns == 0 || pair.incremental_refresh_ns >= pair.full_ns {\n            return EngineChoice::Full;\n        }\n        samples = match samples.checked_add(1) {
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
    fn threshold_is_strict_and_invalid_policy_is_rejected() {
        let target = Geometry { gates: 100, cone: 10 };
        let valid = [pair(12, 80), pair(20, 80)];
        assert_eq!(choose_for_geometry(&valid, target, 2, 2, 20), EngineChoice::Full);
        assert_eq!(choose_for_geometry(&valid, target, 2, 2, 19), EngineChoice::Incremental);
        assert_eq!(choose_for_geometry(&valid, target, 1, 2, 19), EngineChoice::Full);
    }
}
