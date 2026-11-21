use anyhow::Result;
use serde_json::{Value, json};
use std::fs;

// Read procfs only at the boundaries of the CPU suite. Individual dispatch
// timings must exclude procfs I/O and JSON serialization.
#[derive(Clone, Copy)]
struct Counters {
    voluntary_switches: u64,
    involuntary_switches: u64,
    minor_faults: u64,
    major_faults: u64,
}

fn field<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    text.lines()
        .filter_map(|line| line.split_once(':'))
        .find(|(k, _)| k.trim() == key)
        .map(|(_, value)| value.trim())
}

fn counters(status: &str, stat: &str) -> Option<Counters> {
    // comm (field 2) may contain spaces and ')'. Fields 10 and 12 are
    // minor and major faults of the task itself, excluding its children.
    let (_, tail) = stat.rsplit_once(')')?;
    let mut fields = tail.split_whitespace();
    let minor_faults = fields.nth(7)?.parse().ok()?;
    let major_faults = fields.nth(1)?.parse().ok()?;
    Some(Counters {
        voluntary_switches: field(status, "voluntary_ctxt_switches")?.parse().ok()?,
        involuntary_switches: field(status, "nonvoluntary_ctxt_switches")?.parse().ok()?,
        minor_faults,
        major_faults,
    })
}

fn read_counters() -> Option<Counters> {
    counters(
        &fs::read_to_string("/proc/thread-self/status").ok()?,
        &fs::read_to_string("/proc/thread-self/stat").ok()?,
    )
}

pub fn report() -> Result<Value> {
    let cpuinfo = fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let status = fs::read_to_string("/proc/thread-self/status").unwrap_or_default();
    let before = read_counters();
    let mut report = crate::scheduler_bench::report()?;
    let delta = before.zip(read_counters()).map(|(a,b)| json!({
        "voluntary_context_switches":b.voluntary_switches.checked_sub(a.voluntary_switches),
        "involuntary_context_switches":b.involuntary_switches.checked_sub(a.involuntary_switches),
        "minor_page_faults":b.minor_faults.checked_sub(a.minor_faults),
        "major_page_faults":b.major_faults.checked_sub(a.major_faults),
    }));
    report["host"] = json!({
        "os":std::env::consts::OS, "arch":std::env::consts::ARCH,
        "kernel":fs::read_to_string("/proc/sys/kernel/osrelease").ok().map(|v| v.trim().to_owned()),
        "cpu_model":field(&cpuinfo, "model name").or_else(|| field(&cpuinfo, "CPU part")),
        "cpus_allowed":field(&status, "Cpus_allowed_list"),
        "suite_thread_counter_deltas":delta,
    });
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_thread_faults_and_switches_even_for_a_complex_comm() {
        let status = "nonvoluntary_ctxt_switches:\t7\nvoluntary_ctxt_switches:\t9";
        let c = counters(status, "12 (name ) with spaces) R 1 2 3 4 5 6 91 92 93 94").unwrap();
        assert_eq!((c.minor_faults, c.major_faults), (91, 93));
        assert_eq!((c.voluntary_switches, c.involuntary_switches), (9, 7));
    }
}
