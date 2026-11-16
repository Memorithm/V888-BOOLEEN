//! Linux process-wide high-water RSS sampling for bounded sequential experiments.
//! This is a measurement utility, not a biological or performance claim.
use std::io;
use std::time::{Duration, Instant};

/// A sampled process-wide high-water resident set size in KiB.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PeakRss {
    pub kib: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParseError {
    Missing,
    Duplicate,
    Malformed,
    Overflow,
}

/// Strictly parse the VmHWM field from Linux /proc/self/status.
/// Reject ambiguous units, duplicates and truncated values.
pub fn parse_vmhwm(status: &str) -> Result<PeakRss, ParseError> {
    let mut result = None;
    for line in status.lines() {
        let Some(rest) = line.strip_prefix("VmHWM:") else {
            continue;
        };
        if result.is_some() {
            return Err(ParseError::Duplicate);
        }
        let mut fields = rest.split_whitespace();
        let number = fields.next().ok_or(ParseError::Malformed)?;
        let unit = fields.next().ok_or(ParseError::Malformed)?;
        if unit != "kB" || fields.next().is_some() || !number.bytes().all(|b| b.is_ascii_digit()) {
            return Err(ParseError::Malformed);
        }
        let kib = number.parse::<u64>().map_err(|_| ParseError::Overflow)?;
        result = Some(PeakRss { kib });
    }
    result.ok_or(ParseError::Missing)
}

/// Read Linux's process-wide high-water RSS. This is NOT an instantaneous or
/// per-intervention allocation measurement, and can only increase per process.
#[cfg(target_os = "linux")]
pub fn process_peak_rss() -> io::Result<PeakRss> {
    let status = std::fs::read_to_string("/proc/self/status")?;
    parse_vmhwm(&status).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{e:?}")))
}

#[cfg(not(target_os = "linux"))]
pub fn process_peak_rss() -> io::Result<PeakRss> {
    Err(io::Error::new(io::ErrorKind::Unsupported, "VmHWM requires Linux"))
}

/// Time one intervention including the caller's complete cache refresh.
/// The callback must perform the full operation; this utility does not infer
/// whether a cache was rebuilt or whether an intervention was accepted.
pub fn time_intervention<T, E>(
    operation: impl FnOnce() -> Result<T, E>,
) -> (Duration, Result<T, E>) {
    let start = Instant::now();
    let result = operation();
    (start.elapsed(), result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_linux_status_without_confusing_vmrss() {
        let status = "Name:\tprobe\nVmRSS:\t2048 kB\nVmHWM:\t4096 kB\n";
        assert_eq!(parse_vmhwm(status), Ok(PeakRss { kib: 4096 }));
    }

    #[test]
    fn rejects_duplicates_and_unknown_units() {
        assert_eq!(parse_vmhwm("VmHWM: 1 kB\nVmHWM: 2 kB"), Err(ParseError::Duplicate));
        assert_eq!(parse_vmhwm("VmHWM: 2 MB"), Err(ParseError::Malformed));
        assert_eq!(parse_vmhwm("VmRSS: 2 kB"), Err(ParseError::Missing));
    }

    #[test]
    fn rejects_overflow_and_trailing_fields() {
        assert_eq!(parse_vmhwm("VmHWM: 18446744073709551616 kB"), Err(ParseError::Overflow));
        assert_eq!(parse_vmhwm("VmHWM: 12 kB extra"), Err(ParseError::Malformed));
    }

    #[test]
    fn times_success_and_failure_without_changing_result() {
        let (_, ok) = time_intervention(|| Ok::<_, ()>(17));
        let (_, err) = time_intervention(|| Err::<(), _>(9));
        assert_eq!(ok, Ok(17));
        assert_eq!(err, Err(9));
    }
}
