//! Experimental region activation counts from explicitly supplied annotations.
//! Labels are caller-provided; no biological identity is inferred.
use crate::BooleanState;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegionError {
    LengthMismatch,
    CountOverflow,
}

/// Counts active and annotated neurons by label, deterministically ordered.
/// `None` means unannotated and is excluded, never guessed.
pub fn activation_by_region(
    state: &BooleanState,
    labels: &[Option<String>],
) -> Result<BTreeMap<String, (u64, u64)>, RegionError> {
    if state.len() != labels.len() {
        return Err(RegionError::LengthMismatch);
    }
    let mut counts = BTreeMap::<String, (u64, u64)>::new();
    for (index, label) in labels.iter().enumerate() {
        if let Some(label) = label {
            let entry = counts.entry(label.clone()).or_default();
            entry.1 = entry.1.checked_add(1).ok_or(RegionError::CountOverflow)?;
            if state.get(index).expect("index bounded by state length") {
                entry.0 = entry.0.checked_add(1).ok_or(RegionError::CountOverflow)?;
            }
        }
    }
    Ok(counts)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn counts_only_explicit_annotations_and_active_bits() {
        let mut state = BooleanState::new(5);
        state.set(0, true).unwrap();
        state.set(2, true).unwrap();
        state.set(4, true).unwrap();
        let labels = [
            Some("zone-b".into()),
            Some("zone-a".into()),
            None,
            Some("zone-a".into()),
            Some("zone-b".into()),
        ];
        let counts = activation_by_region(&state, &labels).unwrap();
        assert_eq!(counts["zone-a"], (0, 2));
        assert_eq!(counts["zone-b"], (2, 2));
        assert_eq!(counts.len(), 2);
    }
    #[test]
    fn refuses_mismatched_annotation_lengths() {
        let state = BooleanState::new(2);
        assert_eq!(
            activation_by_region(&state, &[None]),
            Err(RegionError::LengthMismatch)
        );
    }
    #[test]
    fn empty_annotations_are_not_invented() {
        assert!(activation_by_region(&BooleanState::new(0), &[])
            .unwrap()
            .is_empty());
    }
}
