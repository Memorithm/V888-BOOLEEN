//! Standalone bounded, streaming audit of sequential synthetic Full/Incremental probes.
//! CSV columns: split,sequence,epoch,gates,cone,full_ns,incremental_ns,refresh_ns,
//! full_output,incremental_output,full_decision,incremental_decision,vm_hwm_kib
//! Rows must be ordered by sequence. This tool never trains on holdouts.
use std::collections::HashSet;
use std::io::{self, BufRead};

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
    equal: bool,
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
        equal: fields[8] == fields[9] && fields[10] == fields[11],
        vm_hwm_kib: number(12)?,
    };
    if obs.gates == 0 || obs.cone > obs.gates || obs.full_ns == 0 || obs.incremental_ns == 0 {
        return Err("invalid geometry or timing".into());
    }
    Ok(obs)
}

#[derive(Default)]
struct Totals { count: u64, mismatch: u64, full: u128, selected: u128, max_hwm: u64 }

fn audit<R: BufRead>(reader: R) -> Result<[Totals; 4], String> {
    let mut totals: [Totals; 4] = std::array::from_fn(|_| Totals::default());
    let mut seen = HashSet::new();
    let mut previous = None;
    for (line_number, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| e.to_string())?;
        if line_number == 0 && line.starts_with("split,") { continue; }
        if line.trim().is_empty() { continue; }
        let o = parse(&line).map_err(|e| format!("line {}: {e}", line_number + 1))?;
        if !seen.insert(o.sequence) { return Err(format!("duplicate sequence {}", o.sequence)); }
        if let Some((seq, epoch)) = previous {
            if o.sequence <= seq || o.epoch < epoch { return Err("non-monotone sequence or cache epoch".into()); }
        }
        previous = Some((o.sequence, o.epoch));
        let idx = match o.split.as_str() { "TRAIN" => 0, "VALID" => 1, "TEST" => 2, _ => 3 };
        let t = &mut totals[idx];
        t.count += 1;
        t.mismatch += u64::from(!o.equal);
        t.full += u128::from(o.full_ns);
        // Conservative retrospective accounting: a mismatch is charged Full.
        // Refresh is charged for every successful Incremental evaluation.
        let incremental_total = u128::from(o.incremental_ns) + u128::from(o.refresh_ns);
        t.selected += if o.equal && incremental_total < u128::from(o.full_ns) {
            incremental_total
        } else { u128::from(o.full_ns) };
        t.max_hwm = t.max_hwm.max(o.vm_hwm_kib);
    }
    Ok(totals)
}

fn main() {
    match audit(io::stdin().lock()) {
        Ok(totals) => {
            println!("split,count,mismatches,full_ns,oracle_safe_lower_bound_ns,vm_hwm_kib");
            for (name, t) in ["TRAIN", "VALID", "TEST", "OOD"].iter().zip(totals.iter()) {
                println!("{name},{},{},{},{},{}", t.count, t.mismatch, t.full, t.selected, t.max_hwm);
            }
            eprintln!("NOTE: oracle_safe_lower_bound_ns is retrospective and uses oracle outcomes; NOT deployable latency or trained policy.");
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
    fn stale_epoch_and_bad_geometry_are_rejected() {
        let csv = format!("{HEADER}VALID,1,2,100,2,100,10,20,1,1,1,1,10\nTEST,2,1,100,2,100,10,20,1,1,1,1,10\n");
        assert!(audit(csv.as_bytes()).is_err());
        assert!(parse("TEST,1,1,100,101,100,10,20,1,1,1,1,10").is_err());
    }
}
