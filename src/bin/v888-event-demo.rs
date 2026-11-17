//! Bounded, reproducible synthetic example of the event-driven Boolean runtime.
use v888_booleen::connectome::Connectome;
use v888_booleen::event_runtime::EventRuntime;

fn main() {
    // Two independently driven sources converge on one threshold-2 target.
    // Source IDs and contacts are synthetic, not a BANC sample.
    let graph = Connectome {
        ids: vec![10, 20, 30],
        offsets: vec![0, 1, 2, 2],
        targets: vec![2, 2],
        counts: vec![1, 1],
    };
    let mut runtime = EventRuntime::new(&graph, 2, 16).expect("valid synthetic graph");
    runtime.schedule_drive(0, 10, true).expect("first drive");
    runtime.schedule_drive(0, 20, true).expect("second drive");

    let sources = runtime.run(2).expect("two source events");
    assert_eq!(sources.state_transitions, 2);
    assert_eq!(sources.wakeups_inserted, 1);
    assert_eq!(sources.wakeups_coalesced, 1);
    assert_eq!(runtime.next_event_time(), Some(1));

    let target = runtime.run(1).expect("one target wakeup");
    assert_eq!(target.threshold_checks, 1);
    assert_eq!(runtime.state().active_count(), 3);
    println!("sources: {sources:?}");
    println!("target: {target:?}");
    println!("active: {}", runtime.state().active_count());
    println!("remaining_events: {}", runtime.pending_count());
    println!("energy_claim: none (logical work counters only)");
}
