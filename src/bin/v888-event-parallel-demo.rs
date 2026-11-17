//! Synthetic deterministic parity demonstration for opt-in concurrent preflight.
//! Spawning worker threads does not establish a physical speedup.

use v888_booleen::connectome::Connectome;
use v888_booleen::event_runtime::EventRuntime;

fn main() {
    // Eight independent stimulated sources converge on a ninth target.
    let graph = Connectome {
        ids: (10..=18).collect(),
        offsets: (0..=8).chain(std::iter::once(8)).collect(),
        targets: vec![8; 8],
        counts: vec![1; 8],
    };
    let mut reference = EventRuntime::new(&graph, 4, 64).expect("valid synthetic graph");
    let mut parallel = EventRuntime::new(&graph, 4, 64).expect("valid synthetic graph");
    for id in 10..18 {
        reference.schedule_drive(0, id, true).expect("valid source");
        parallel.schedule_drive(0, id, true).expect("valid source");
    }
    let sequential_work = reference.run(64).expect("reference events");
    let parallel_report = parallel
        .run_parallel_frontiers(64, 4)
        .expect("bounded concurrent preflight");
    assert_eq!(sequential_work, parallel_report.work);
    assert_eq!(reference.state(), parallel.state());
    assert_eq!(reference.incoming_for(18), parallel.incoming_for(18));
    assert_eq!(parallel.audit_incoming(), Ok(true));
    assert_eq!(parallel.pending_count(), reference.pending_count());

    println!("logical_work_equal=true");
    println!("parallel_groups={}", parallel_report.parallel_groups);
    println!("prepared_events={}", parallel_report.prepared_events);
    println!("threads_spawned={}", parallel_report.threads_spawned);
    println!("physical_speedup_claimed=false");
}
