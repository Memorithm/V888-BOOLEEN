//! Experimental activity-guided circuit selection on BANC V888.
//! This constructs a *lossy, derived* induced subgraph, never an anatomical
//! source replacement. Observed activation means simulated, not biological.
use crate::activity_atlas::ActivityAtlas;
use crate::connectome::Connectome;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::io::{self, ErrorKind};

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, message)
}

#[derive(Debug, Clone, Copy)]
pub struct SelectionConfig {
    /// Up to this many most frequently active neurons become seed anchors.
    pub anchors: usize,
    /// Upper bound on vertices in the derived circuit (including anchors).
    pub max_neurons: usize,
    /// An anchor must have appeared in at least this many observed ticks.
    pub min_active_ticks: u32,
    /// Directed, positive-contact expansion from anchors (0..=3).
    pub expansion_hops: u8,
    /// Strongest outgoing rows considered per expanded neuron (1..=64).
    pub fanout: usize,
    /// Exact budget for inspected source rows during expansion and projection.
    pub max_scanned_edges: u64,
    /// Maximum edge rows in output. Never silently truncate rows.
    pub max_result_edges: usize,
}

impl Default for SelectionConfig {
    fn default() -> Self {
        Self {
            anchors: 128,
            max_neurons: 2048,
            min_active_ticks: 1,
            expansion_hops: 1,
            fanout: 8,
            max_scanned_edges: 20_000_000,
            max_result_edges: 1_000_000,
        }
    }
}

impl SelectionConfig {
    pub fn validate(self) -> io::Result<()> {
        if self.anchors == 0
            || self.anchors > 10_000
            || self.max_neurons == 0
            || self.max_neurons > 10_000
            || self.anchors > self.max_neurons
            || self.min_active_ticks == 0
            || self.expansion_hops > 3
            || !(1..=64).contains(&self.fanout)
            || self.max_scanned_edges == 0
            || self.max_scanned_edges > 100_000_000
            || self.max_result_edges == 0
            || self.max_result_edges > 2_000_000
        {
            return Err(invalid("invalid activity-guided selection budget"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectionReport {
    pub recorded_ticks: u32,
    pub anchors: usize,
    pub expanded_neurons: usize,
    pub graph_neurons: usize,
    pub graph_edges: usize,
    pub selected_neurons: usize,
    pub selected_edges: usize,
    pub inspected_edges: u64,
}

fn charge(scanned: &mut u64, limit: u64) -> io::Result<()> {
    if *scanned >= limit {
        return Err(invalid("edge scanning budget exceeded; no partial circuit returned"));
    }
    *scanned += 1;
    Ok(())
}

/// Extract a deterministic, activity-guided directed circuit.
///
/// 1. Take frequent simulated activations as anchors.
/// 2. Optionally expand over the strongest existing directed outgoing rows.
/// 3. Preserve *every* original row between selected vertices, including
///    zero contacts, duplicate rows and self-loops.
///
/// Expansion is a routing hypothesis, NOT anatomical discovery. The original
/// graph and its BANC provenance must remain available separately.
pub fn derive_circuit(
    graph: &Connectome,
    atlas: &ActivityAtlas,
    config: SelectionConfig,
) -> io::Result<(Connectome, SelectionReport)> {
    config.validate()?;
    graph.validate()?;
    let ranked = atlas.top_neurons(graph, config.anchors)?;
    let anchors: Vec<usize> = ranked
        .iter()
        .filter(|activity| activity.active_ticks >= config.min_active_ticks)
        .map(|activity| {
            graph
                .index(activity.neuron_id)
                .ok_or_else(|| invalid("atlas neuron absent from connectome"))
        })
        .collect::<io::Result<Vec<usize>>>()?;
    if anchors.is_empty() {
        return Err(invalid("no observed neurons meet the anchor threshold"));
    }

    let mut chosen = vec![false; graph.ids.len()];
    let mut selected_count = 0_usize;
    for &index in &anchors {
        if !chosen[index] {
            chosen[index] = true;
            selected_count += 1;
        }
    }
    let mut frontier = anchors.clone();
    let mut scanned_edges = 0_u64;

    for _ in 0..config.expansion_hops {
        if selected_count == config.max_neurons || frontier.is_empty() {
            break;
        }
        let mut next = Vec::new();
        for &source in &frontier {
            if selected_count == config.max_neurons {
                break;
            }
            // Heap holds at most fanout rows. Largest count wins; ties prefer
            // lower target index, then earlier original row index.
            type Score = (u64, Reverse<u32>, Reverse<usize>);
            let mut strongest: BinaryHeap<Reverse<Score>> = BinaryHeap::new();
            let start = graph.offsets[source] as usize;
            let end = graph.offsets[source + 1] as usize;
            for edge in start..end {
                charge(&mut scanned_edges, config.max_scanned_edges)?;
                let contacts = graph.counts[edge];
                if contacts == 0 {
                    continue;
                }
                let score = (contacts, Reverse(graph.targets[edge]), Reverse(edge));
                if strongest.len() < config.fanout {
                    strongest.push(Reverse(score));
                } else if strongest.peek().is_some_and(|worst| score > worst.0) {
                    strongest.pop();
                    strongest.push(Reverse(score));
                }
            }
            let mut ordered: Vec<Score> =
                strongest.into_iter().map(|Reverse(score)| score).collect();
            ordered.sort_unstable_by(|a, b| b.cmp(a));
            for (_, Reverse(target), _) in ordered {
                let target = target as usize;
                if !chosen[target] && selected_count < config.max_neurons {
                    chosen[target] = true;
                    selected_count += 1;
                    next.push(target);
                }
            }
        }
        frontier = next;
    }

    // Root IDs remain sorted, while original duplicate and zero-count edge
    // rows survive intact when both endpoints are selected.
    let mut remap = vec![usize::MAX; graph.ids.len()];
    let mut ids = Vec::with_capacity(selected_count);
    for (index, &selected) in chosen.iter().enumerate() {
        if selected {
            remap[index] = ids.len();
            ids.push(graph.ids[index]);
        }
    }
    let mut offsets = Vec::with_capacity(ids.len() + 1);
    let mut targets = Vec::new();
    let mut counts = Vec::new();
    offsets.push(0_u64);
    for (source, &selected) in chosen.iter().enumerate() {
        if !selected {
            continue;
        }
        for edge in graph.offsets[source] as usize..graph.offsets[source + 1] as usize {
            charge(&mut scanned_edges, config.max_scanned_edges)?;
            let target = remap[graph.targets[edge] as usize];
            if target == usize::MAX {
                continue;
            }
            if targets.len() == config.max_result_edges {
                return Err(invalid("projection edge budget exceeded; no partial circuit returned"));
            }
            targets.push(u32::try_from(target).map_err(|_| invalid("target index overflow"))?);
            counts.push(graph.counts[edge]);
        }
        offsets.push(targets.len() as u64);
    }

    let derived = Connectome {
        ids,
        offsets,
        targets,
        counts,
    };
    derived.validate()?;
    let report = SelectionReport {
        recorded_ticks: atlas.ticks(),
        anchors: anchors.len(),
        expanded_neurons: selected_count - anchors.len(),
        graph_neurons: graph.ids.len(),
        graph_edges: graph.edges(),
        selected_neurons: derived.ids.len(),
        selected_edges: derived.edges(),
        inspected_edges: scanned_edges,
    };
    Ok((derived, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BooleanState;

    fn sample() -> Connectome {
        Connectome {
            ids: vec![10, 20, 30, 40, 50, 60],
            offsets: vec![0, 3, 4, 5, 6, 6, 6],
            targets: vec![1, 1, 2, 3, 4, 5],
            counts: vec![3, 0, 2, 1, 1, 1],
        }
    }

    fn atlas_with_activations(graph: &Connectome) -> ActivityAtlas {
        let mut atlas = ActivityAtlas::new(graph).unwrap();
        let mut active = BooleanState::new(graph.ids.len());
        active.set(0, true).unwrap();
        active.set(1, true).unwrap();
        atlas.observe(graph, &active).unwrap();
        active.set(1, false).unwrap();
        atlas.observe(graph, &active).unwrap();
        atlas
    }

    fn config() -> SelectionConfig {
        SelectionConfig {
            anchors: 1,
            max_neurons: 3,
            min_active_ticks: 1,
            expansion_hops: 1,
            fanout: 2,
            max_scanned_edges: 20,
            max_result_edges: 10,
        }
    }

    #[test]
    fn projects_existing_rows_including_duplicates_and_zero_counts() {
        let graph = sample();
        let before = graph.clone();
        let atlas = atlas_with_activations(&graph);
        let (reduced, r) = derive_circuit(&graph, &atlas, config()).unwrap();
        assert_eq!(reduced.ids, vec![10, 20, 30]);
        assert_eq!(reduced.offsets, vec![0, 3, 3, 3]);
        assert_eq!(reduced.targets, vec![1, 1, 2]);
        assert_eq!(reduced.counts, vec![3, 0, 2]);
        assert_eq!(r.anchors, 1);
        assert_eq!(r.expanded_neurons, 2);
        assert_eq!(r.inspected_edges, 8); // 3 expanded, then 3+1+1 projected
        assert_eq!(graph, before);
    }

    #[test]
    fn different_observations_adapt_circuit_selection() {
        let graph = sample();
        let atlas_10 = atlas_with_activations(&graph);
        let mut atlas_20 = ActivityAtlas::new(&graph).unwrap();
        let mut active = BooleanState::new(graph.ids.len());
        active.set(1, true).unwrap();
        atlas_20.observe(&graph, &active).unwrap();
        atlas_20.observe(&graph, &active).unwrap();
        let cfg = SelectionConfig {
            max_neurons: 2,
            fanout: 1,
            ..config()
        };
        let (a, _) = derive_circuit(&graph, &atlas_10, cfg).unwrap();
        let (b, _) = derive_circuit(&graph, &atlas_20, cfg).unwrap();
        assert_eq!(a.ids, vec![10, 20]);
        assert_eq!(b.ids, vec![20, 40]);
        assert_eq!(b.counts, vec![1]);
    }

    #[test]
    fn derived_graph_is_reopenable_and_original_file_cannot_be_clobbered() {
        let original = sample();
        let atlas = atlas_with_activations(&original);
        let (derived, _) = derive_circuit(&original, &atlas, config()).unwrap();
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "v888-derived-{}-{unique}.v888b",
            std::process::id()
        ));
        derived.save_new(&path).unwrap();
        assert_eq!(Connectome::load(&path).unwrap(), derived);
        assert!(derived.save_new(&path).is_err());
        assert_eq!(Connectome::load(&path).unwrap(), derived);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn fails_closed_on_budget_and_wrong_graph() {
        let graph = sample();
        let atlas = atlas_with_activations(&graph);
        let mut budget = config();
        budget.max_scanned_edges = 3;
        assert!(derive_circuit(&graph, &atlas, budget).is_err());
        budget = config();
        budget.max_result_edges = 2;
        assert!(derive_circuit(&graph, &atlas, budget).is_err());
        let mut another = graph.clone();
        another.counts[0] += 1;
        assert!(derive_circuit(&another, &atlas, config()).is_err());
    }
}
