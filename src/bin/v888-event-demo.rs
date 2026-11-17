//! Bounded, reproducible synthetic example of the event-driven Boolean runtime.
use v888_booleen::connectome::Connectome;
use v888_booleen::event_runtime::EventRuntime;
use v888_booleen::event_resources::{EventResourceTariffs, EventResourceVector};

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

    // A budget of zero cannot admit the first transition, but no
    // state change or incoming contribution is lost.
    let deferred = runtime
        .run_with_contact_budget(2, 0)
        .expect("bounded preflight");
    assert!(deferred.blocked_on_contact_budget);
    assert_eq!(deferred.work.events_consumed, 0);

    let admitted = runtime
        .run_with_contact_budget(2, 2)
        .expect("two source events");
    let sources = admitted.work;
    assert_eq!(sources.state_transitions, 2);
    assert_eq!(sources.wakeups_inserted, 1);
    assert_eq!(sources.wakeups_coalesced, 1);
    assert_eq!(runtime.next_event_time(), Some(1));

    let target = runtime.run(1).expect("one target wakeup");
    assert_eq!(target.threshold_checks, 1);
    assert_eq!(runtime.state().active_count(), 3);
    let resource_vector = EventResourceVector::from_work(sources)
        .checked_add(EventResourceVector::from_work(target))
        .expect("synthetic resource count fits u128");
    let arbitrary_tariffs = EventResourceTariffs {
        coefficients: [1; 7],
    };
    let proxy = resource_vector
        .checked_proxy_score(arbitrary_tariffs)
        .expect("synthetic proxy score fits u128");
    println!("arbitrary_unit_proxy_score: {proxy}");
    println!("sources: {sources:?}");
    println!("target: {target:?}");
    println!("active: {}", runtime.state().active_count());
    println!("remaining_events: {}", runtime.pending_count());
    println!("energy_claim: none (logical work counters only)");
}
