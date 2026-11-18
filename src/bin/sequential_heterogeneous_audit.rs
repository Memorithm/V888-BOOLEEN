//! Standalone bounded, streaming audit of sequential synthetic Full/Incremental probes.
//! CSV columns: split,sequence,epoch,gates,cone,full_ns,incremental_ns,refresh_ns,
//! full_output,incremental_output,full_decision,incremental_decision,vm_hwm_kib
//! Rows must be ordered by sequence. This tool never trains on holdouts.
use std::io::{self, BufRead, Read};
use std::time::Instant;

// Reject oversized CSV records before allocating unbounded input memory.
const MAX_RECORD_BYTES: u64 = 4096;


// Physical process peak RSS on Linux, independent of input-reported vm_hwm_kib.
// A missing /proc or malformed status is reported as unavailable, never as zero.
fn parse_process_vm_hwm(status: &str) -> Option<u64> {
    let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
    let mut parts = line["VmHWM:".len()..].split_whitespace();
    let kib = parts.next()?.parse::<u64>().ok()?;
    if parts.next()? != "kB" || parts.next().is_some() {
        return None;
    }
    Some(kib)
}

fn physical_process_vm_hwm_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    parse_process_vm_hwm(&status)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Observation {
    split: String,
    sequence: u64,
    epoch: u64,
    gates: u64,
    cone: u64,
    full_ns: u64,
    incremental_ns: u64,
    refresh_ns: u64,
    output_equal: bool,
    decision_equal: bool,
    vm_hwm_kib: u64,
}

fn parse(line: &str) -> Result<Observation, String> {
    let fields: Vec<_> = line.trim_end().split(',').collect();
    if fields.len() != 13 { return Err("expected 13 columns".into()); }
    let number = |i: usize| fields[i].parse::<u64>().map_err(|_| format!("invalid integer column {i}"));
    if !matches!(fields[0], "TRAIN" | "VALID" | "TEST" | "OOD") { return Err("invalid split".into()); }
    let obs = Observation {
        split: fields[0].into(), sequence: number(1)?, epoch: number(2)?,
        gates: number(3)?, cone: number(4)?, full_ns: number(5)?,
        incremental_ns: number(6)?, refresh_ns: number(7)?,
        output_equal: fields[8] == fields[9],
        decision_equal: fields[10] == fields[11],
        vm_hwm_kib: number(12)?,
    };
    if !matches!(fields[8], "0" | "1") || !matches!(fields[9], "0" | "1")
        || !matches!(fields[10], "0" | "1") || !matches!(fields[11], "0" | "1") {
        return Err("invalid Boolean output or decision".into());
    }
    if obs.gates == 0 || obs.cone > obs.gates || obs.full_ns == 0 || obs.incremental_ns == 0 {
        return Err("invalid geometry or timing".into());
    }
    Ok(obs)
}

#[derive(Default)]
struct Totals { count: u64, mismatch: u64, output_mismatch: u64, decision_mismatch: u64, full: u128, selected: u128, incremental: u128, refresh: u128, epoch_transitions: u64, max_hwm: u64, geometry_transitions: u64, min_cone: Option<u64>, max_cone: u64, max_refresh_ns: u64 }

fn audit<R: BufRead>(mut reader: R) -> Result<[Totals; 4], String> {
    let mut totals: [Totals; 4] = std::array::from_fn(|_| Totals::default());
    let mut previous = None;
    let mut previous_epoch_by_split: [Option<u64>; 4] = [None; 4];
    let mut previous_geometry_by_split: [Option<(u64, u64)>; 4] = [None; 4];
    let mut line_number = 0usize;
    loop {
        let mut bytes = Vec::new();
        let read = reader.by_ref().take(MAX_RECORD_BYTES + 1).read_until(b'\n', &mut bytes)
            .map_err(|e| e.to_string())?;
        if read == 0 { break; }
        line_number += 1;
        if read as u64 > MAX_RECORD_BYTES { return Err(format!("line {line_number}: record exceeds 4096 bytes")); }
        let line = std::str::from_utf8(&bytes).map_err(|_| format!("line {line_number}: invalid UTF-8"))?;
        if line_number == 1 && line.starts_with("split,") { continue; }
        if line.trim().is_empty() { continue; }
        let o = parse(&line).map_err(|e| format!("line {}: {e}", line_number))?;
        if let Some((seq, epoch)) = previous {
            if o.sequence <= seq || o.epoch < epoch { return Err("non-monotone sequence or cache epoch".into()); }
        }
        previous = Some((o.sequence, o.epoch));
        let idx = match o.split.as_str() { "TRAIN" => 0, "VALID" => 1, "TEST" => 2, _ => 3 };
        let t = &mut totals[idx];
        t.count = t.count.checked_add(1).ok_or("count overflow")?;
        if let Some(old_epoch) = previous_epoch_by_split[idx] {
            if old_epoch != o.epoch {
                t.epoch_transitions = t.epoch_transitions.checked_add(1).ok_or("epoch transition overflow")?;
            }
        }
        previous_epoch_by_split[idx] = Some(o.epoch);
        if let Some(old_geometry) = previous_geometry_by_split[idx] {
            if old_geometry != (o.gates, o.cone) {
                t.geometry_transitions = t.geometry_transitions.checked_add(1).ok_or("geometry transition overflow")?;
            }
        }
        previous_geometry_by_split[idx] = Some((o.gates, o.cone));
        t.min_cone = Some(t.min_cone.map_or(o.cone, |min| min.min(o.cone)));
        t.max_cone = t.max_cone.max(o.cone);
        t.max_refresh_ns = t.max_refresh_ns.max(o.refresh_ns);
        t.incremental = t.incremental.checked_add(u128::from(o.incremental_ns)).ok_or("incremental timing overflow")?;
        t.refresh = t.refresh.checked_add(u128::from(o.refresh_ns)).ok_or("refresh timing overflow")?;
        t.output_mismatch = t.output_mismatch.checked_add(u64::from(!o.output_equal)).ok_or("output mismatch overflow")?;
        t.decision_mismatch = t.decision_mismatch.checked_add(u64::from(!o.decision_equal)).ok_or("decision mismatch overflow")?;
        let equal = o.output_equal && o.decision_equal;
        t.mismatch = t.mismatch.checked_add(u64::from(!equal)).ok_or("mismatch overflow")?;
        t.full = t.full.checked_add(u128::from(o.full_ns)).ok_or("full timing overflow")?;
        // Conservative retrospective accounting: a mismatch is charged Full.
        // Refresh is charged for every successful Incremental evaluation.
        let incremental_total = u128::from(o.incremental_ns) + u128::from(o.refresh_ns);
        let selected = if equal && incremental_total < u128::from(o.full_ns) {
            incremental_total
        } else { u128::from(o.full_ns) };
        t.selected = t.selected.checked_add(selected).ok_or("selected timing overflow")?;
        t.max_hwm = t.max_hwm.max(o.vm_hwm_kib);
    }
    Ok(totals)
}

fn main() {
    // This clock measures the audit process only, never the Boolean engine.
    let audit_start = Instant::now();
    match audit(io::stdin().lock()) {
        Ok(totals) => {
            let audit_elapsed_ns = audit_start.elapsed().as_nanos();
            let total_rows: u128 = totals.iter().map(|t| u128::from(t.count)).sum();
            let audit_rows_per_second = if audit_elapsed_ns == 0 { None } else {
                total_rows.checked_mul(1_000_000_000).map(|n| n / audit_elapsed_ns)
            };
            eprintln!("PHYSICAL_AUDITOR_ELAPSED_NS={audit_elapsed_ns}");
            match audit_rows_per_second {
                Some(rate) => eprintln!("PHYSICAL_AUDITOR_ROWS_PER_SECOND={rate}"),
                None => eprintln!("PHYSICAL_AUDITOR_ROWS_PER_SECOND=unavailable"),
            }
            println!("split,count,mismatches,output_mismatches,decision_mismatches,full_ns,oracle_safe_lower_bound_ns,incremental_ns,refresh_ns,epoch_transitions,oracle_bound_ops_per_second,vm_hwm_kib,geometry_transitions,min_cone,max_cone,max_refresh_ns");
            for (name, t) in ["TRAIN", "VALID", "TEST", "OOD"].iter().zip(totals.iter()) {
                let throughput = if t.selected == 0 { 0 } else { u128::from(t.count).saturating_mul(1_000_000_000) / t.selected };
                println!("{name},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}", t.count, t.mismatch, t.output_mismatch, t.decision_mismatch, t.full, t.selected, t.incremental, t.refresh, t.epoch_transitions, throughput, t.max_hwm, t.geometry_transitions, t.min_cone.unwrap_or(0), t.max_cone, t.max_refresh_ns);
            }
            match physical_process_vm_hwm_kib() {
                Some(kib) => eprintln!("PHYSICAL_AUDITOR_PROCESS_VM_HWM_KIB={kib}"),
                None => eprintln!("PHYSICAL_AUDITOR_PROCESS_VM_HWM_KIB=unavailable"),
            }
            eprintln!("NOTE: PHYSICAL_AUDITOR_* measures only CSV ingestion and accounting, NOT engine throughput or end-to-end inference. oracle_safe_lower_bound_ns and oracle_bound_ops_per_second are retrospective optimistic bounds using oracle outcomes, NOT measured wall-clock throughput, deployable latency or trained policy. incremental_ns and refresh_ns are input-reported sums, not physical measurements made by this program.");
        }
        Err(e) => { eprintln!("audit rejected: {e}"); std::process::exit(2); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const HEADER: &str = "split,sequence,epoch,gates,cone,full_ns,incremental_ns,refresh_ns,full_output,incremental_output,full_decision,incremental_decision,vm_hwm_kib\n";
    #[test]
    fn independent_oracle_rejects_mismatch_and_charges_refresh() {
        let csv = format!("{HEADER}VALID,1,1,100,2,100,10,20,1,1,1,1,1024\nTEST,2,2,100,3,100,10,20,1,0,1,1,2048\n");
        let t = audit(csv.as_bytes()).unwrap();
        assert_eq!((t[1].selected, t[1].mismatch), (30, 0));
        assert_eq!((t[2].selected, t[2].mismatch, t[2].max_hwm), (100, 1, 2048));
    }
    #[test]
    fn duplicate_across_splits_is_rejected() {
        let csv = format!("{HEADER}VALID,1,1,100,2,100,10,20,1,1,1,1,10\nTEST,1,2,100,2,100,10,20,1,1,1,1,10\n");
        assert!(audit(csv.as_bytes()).is_err());
    }
    #[test]
    fn invalid_boolean_is_not_silently_counted_as_oracle_agreement() {
        assert!(parse("VALID,1,1,100,2,100,10,20,unknown,unknown,1,1,10").is_err());
        assert!(parse("VALID,1,1,100,2,100,10,20,1,1,2,2,10").is_err());
    }
    #[test]
    fn ordered_stream_requires_no_unbounded_sequence_set() {
        let csv = format!("{HEADER}TRAIN,1,0,10,1,100,10,0,0,0,1,1,10\nVALID,2,0,10,1,100,10,0,0,0,1,1,10\n");
        let t = audit(csv.as_bytes()).unwrap();
        assert_eq!((t[0].count, t[1].count), (1, 1));
        assert!(audit(format!("{HEADER}TRAIN,2,0,10,1,100,10,0,0,0,1,1,10\nVALID,1,0,10,1,100,10,0,0,0,1,1,10\n").as_bytes()).is_err());
    }
    #[test]
    fn reports_split_local_epoch_transitions_and_input_costs() {
        let csv = format!("{HEADER}TRAIN,1,0,100,2,100,10,20,1,1,1,1,10\nVALID,2,1,100,2,100,12,3,1,1,1,1,20\nTRAIN,3,1,100,2,100,14,4,1,1,1,1,30\nTRAIN,4,1,100,2,100,16,5,1,1,1,1,40\n");
        let t = audit(csv.as_bytes()).unwrap();
        assert_eq!((t[0].count, t[0].epoch_transitions, t[0].incremental, t[0].refresh), (3, 1, 40, 29));
        assert_eq!((t[1].count, t[1].epoch_transitions, t[1].incremental, t[1].refresh), (1, 0, 12, 3));
        assert_eq!(t[0].max_hwm, 40);
    }
    #[test]
    fn heterogeneous_geometry_is_tracked_without_cross_split_contamination() {
        let csv = format!("{HEADER}TRAIN,1,0,100,2,100,10,20,1,1,1,1,10\nVALID,2,0,100,9,100,10,30,1,1,1,1,20\nTRAIN,3,1,100,2,100,10,40,1,1,1,1,30\nTRAIN,4,1,200,7,100,10,50,1,1,1,1,40\n");
        let t = audit(csv.as_bytes()).unwrap();
        assert_eq!((t[0].geometry_transitions, t[0].min_cone, t[0].max_cone, t[0].max_refresh_ns), (1, Some(2), 7, 50));
        assert_eq!((t[1].geometry_transitions, t[1].min_cone, t[1].max_cone, t[1].max_refresh_ns), (0, Some(9), 9, 30));
        assert_eq!((t[2].geometry_transitions, t[2].min_cone), (0, None));
    }
    #[test]
    fn oversized_record_is_rejected_without_unbounded_allocation() {
        let csv = format!("{HEADER}{}\n", "x".repeat(5000));
        assert!(audit(csv.as_bytes()).unwrap_err().contains("record exceeds"));
    }
    #[test]
    fn invalid_utf8_record_is_rejected() {
        let mut csv = HEADER.as_bytes().to_vec();
        csv.extend_from_slice(&[0xff, b'\n']);
        assert!(audit(csv.as_slice()).unwrap_err().contains("invalid UTF-8"));
    }
    #[test]
    fn independent_oracle_separates_output_and_decision_disagreements() {
        let csv = format!("{HEADER}VALID,1,0,100,2,100,10,5,1,0,1,1,10\nVALID,2,0,100,2,100,10,5,1,1,1,0,10\nVALID,3,0,100,2,100,10,5,1,0,1,0,10\nVALID,4,0,100,2,100,10,5,1,1,1,1,10\n");
        let t = audit(csv.as_bytes()).unwrap();
        assert_eq!((t[1].count, t[1].output_mismatch, t[1].decision_mismatch, t[1].mismatch), (4, 2, 2, 3));
        assert_eq!(t[1].selected, 315); // Three rejected probes use Full; only the agreement uses Incremental + refresh.
        assert_eq!(t[0].mismatch, 0); // Holdout split counters never leak into TRAIN.
    }
    #[test]
    fn stale_epoch_and_bad_geometry_are_rejected() {
        let csv = format!("{HEADER}VALID,1,2,100,2,100,10,20,1,1,1,1,10\nTEST,2,1,100,2,100,10,20,1,1,1,1,10\n");
        assert!(audit(csv.as_bytes()).is_err());
        assert!(parse("TEST,1,1,100,101,100,10,20,1,1,1,1,10").is_err());
    }
}
