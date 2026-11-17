//! Bounded synthetic demonstration of in-place disjoint counter commits.
//! Does not claim faster runtime or power efficiency.
use v888_booleen::connectome::Connectome;
use v888_booleen::event_runtime::EventRuntime;

fn main() {
    let graph = Connectome {
        ids: (10..26).collect(),
        offsets: (0..=8).chain(std::iter::repeat_n(8, 8)).collect(),
        targets: (8..16).collect(),
        counts: vec![1; 8],
    };
    let mut baseline = EventRuntime::new(&graph, 1, 64).expect("valid synthetic graph");
    let mut disjoint = EventRuntime::new(&graph, 1, 64).expect("valid synthetic graph");
    for id in 10..18 {
        baseline.schedule_drive(0, id, true).expect("valid drive");
        disjoint.schedule_drive(0, id, true).expect("valid drive");
    }
    let expected = baseline.run(24).expect("serial reference");
    let actual = disjoint
        .run_disjoint_parallel_commits(24, 4)
        .expect("parallel disjoint commits");
    assert_eq!(actual.work, expected);
    assert_eq!(disjoint.state(), baseline.state());
    assert_eq!(disjoint.pending_count(), baseline.pending_count());
    assert_eq!(disjoint.audit_incoming(), Ok(true));
    assert_eq!(actual.disjoint_groups, 1);
    assert_eq!(actual.target_counters_written, 8);
    assert_eq!(actual.target_writer_threads, 4);
    println!("exact_logical_parity=true");
    println!("disjoint_groups={}", actual.disjoint_groups);
    println!("parallel_counter_writes={}", actual.target_counters_written);
    println!("writer_threads_spawned={}", actual.target_writer_threads);
    println!("physical_speedup_claimed=false");
}
