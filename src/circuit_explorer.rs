//! Experimental circuit routing on the preserved V888 directed graph.
//! Routing choices are hypotheses, not evidence of biological dynamics.
//! The source Connectome is never rewritten; virtual reverse edges are runtime-only.
use crate::connectome::Connectome;
use crate::gates;
use crate::BooleanState;
use std::collections::BTreeMap;
use std::io::{self, BufRead, ErrorKind};

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, message)
}

/// Three experimentally distinct ways to route activity.
/// Directed preserves every source row; StrongestK retains only the highest
/// contact-count outgoing rows per source (ties: target index, then row index);
/// SymmetricHypothesis adds reversed links in memory without changing anatomy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Routing {
    Directed,
    StrongestK(usize),
    SymmetricHypothesis,
}

impl Routing {
    pub fn parse(value: &str, fanout: usize) -> io::Result<Self> {
        match value {
            "directed" => Ok(Self::Directed),
            "strongest-k" if (1..=1024).contains(&fanout) => Ok(Self::StrongestK(fanout)),
            "symmetric-hypothesis" => Ok(Self::SymmetricHypothesis),
            _ => Err(invalid(
                "unknown routing policy or invalid strongest-k fanout",
            )),
        }
    }
}

/// Optional labels from an externally supplied neuron_id,region CSV.
/// Unannotated neurons are explicitly UNMAPPED, never assigned an invented region.
#[derive(Clone, Debug)]
pub struct RegionLabels {
    labels: Vec<String>,
    graph_ids: Vec<u64>,
}

impl RegionLabels {
    pub fn from_reader<R: BufRead>(graph: &Connectome, source: R) -> io::Result<Self> {
        graph.validate()?;
        let mut labels = vec![String::from("UNMAPPED"); graph.ids.len()];
        let mut seen = vec![false; graph.ids.len()];
        let mut rows = source.lines();
        let header = rows
            .next()
            .ok_or_else(|| invalid("missing region CSV header"))??;
        if header.trim_end_matches('\r') != "neuron_id,region" {
            return Err(invalid("region CSV requires neuron_id,region header"));
        }
        for row in rows {
            let row = row?;
            let (id, label) = row
                .trim_end_matches('\r')
                .split_once(',')
                .ok_or_else(|| invalid("invalid region CSV row"))?;
            if label.is_empty()
                || label.len() > 64
                || label.contains(',')
                || !label
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
            {
                return Err(invalid("invalid region name"));
            }
            let id = id
                .parse::<u64>()
                .map_err(|_| invalid("invalid neuron ID"))?;
            let index = graph
                .index(id)
                .ok_or_else(|| invalid("unknown neuron ID in regions"))?;
            if seen[index] {
                return Err(invalid("duplicate neuron ID in regions"));
            }
            seen[index] = true;
            labels[index] = label.to_owned();
        }
        Ok(Self {
            labels,
            graph_ids: graph.ids.clone(),
        })
    }

    fn label(&self, index: usize) -> &str {
        &self.labels[index]
    }
}

/// Observable circuit activity, including a bounded sample of original neuron IDs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActivityReport {
    pub active_neurons: u64,
    pub newly_active: u64,
    pub silenced: u64,
    pub active_sources_visited: u64,
    pub routed_edges: u64,
    pub contact_mass: u128,
    pub active_by_region: BTreeMap<String, u64>,
    pub sampled_neuron_ids: Vec<u64>,
}

/// Prepares routing once, reuses the index at every synchronous recurrent step.
/// StrongestK caps the number of outgoing rows *read during a step* but its
/// preparation still scans the original rows; no zero-cost-memory-read claim.
pub struct CircuitExplorer<'a> {
    graph: &'a Connectome,
    pub active: BooleanState,
    threshold: u128,
    selected: Option<(Vec<usize>, Vec<usize>)>,
    reverse: Option<Vec<Vec<(usize, u64)>>>,
}

impl<'a> CircuitExplorer<'a> {
    pub fn new(graph: &'a Connectome, routing: Routing, threshold: u128) -> io::Result<Self> {
        graph.validate()?;
        if threshold == 0 {
            return Err(invalid("threshold must be positive"));
        }
        let selected = if let Routing::StrongestK(k) = routing {
            if !(1..=1024).contains(&k) {
                return Err(invalid("fanout must be in 1..=1024"));
            }
            let mut offsets = Vec::with_capacity(graph.ids.len() + 1);
            let mut edges = Vec::new();
            offsets.push(0);
            for source in 0..graph.ids.len() {
                let start = graph.offsets[source] as usize;
                let end = graph.offsets[source + 1] as usize;
                let mut candidates: Vec<usize> = (start..end).collect();
                candidates.sort_unstable_by(|a, b| {
                    graph.counts[*b]
                        .cmp(&graph.counts[*a])
                        .then_with(|| graph.targets[*a].cmp(&graph.targets[*b]))
                        .then_with(|| a.cmp(b))
                });
                candidates.truncate(k);
                edges.extend(candidates);
                offsets.push(edges.len());
            }
            Some((offsets, edges))
        } else {
            None
        };

        // Cap an optional, memory-expensive experimental reverse index.
        let reverse = if routing == Routing::SymmetricHypothesis {
            if graph.edges() > 30_000_000 {
                return Err(invalid("reverse routing exceeds its edge memory budget"));
            }
            let mut incoming = vec![Vec::<(usize, u64)>::new(); graph.ids.len()];
            for source in 0..graph.ids.len() {
                for edge in graph.offsets[source] as usize..graph.offsets[source + 1] as usize {
                    let target = graph.targets[edge] as usize;
                    if target != source {
                        incoming[target].push((source, graph.counts[edge]));
                    }
                }
            }
            Some(incoming)
        } else {
            None
        };

        Ok(Self {
            graph,
            active: BooleanState::new(graph.ids.len()),
            threshold,
            selected,
            reverse,
        })
    }

    pub fn stimulate(&mut self, neuron_id: u64) -> io::Result<()> {
        if self.active.len() != self.graph.ids.len() {
            return Err(invalid("graph/state size mismatch"));
        }
        let index = self
            .graph
            .index(neuron_id)
            .ok_or_else(|| invalid("unknown seed neuron"))?;
        self.active
            .set(index, true)
            .map_err(|_| invalid("invalid neuron index"))
    }

    pub fn snapshot(&self, regions: Option<&RegionLabels>) -> io::Result<ActivityReport> {
        if self.active.len() != self.graph.ids.len() {
            return Err(invalid("graph/state size mismatch"));
        }
        if regions.is_some_and(|r| r.graph_ids.as_slice() != self.graph.ids.as_slice()) {
            return Err(invalid("region label neuron IDs do not match this graph"));
        }
        let mut active_by_region = BTreeMap::new();
        let mut sampled_neuron_ids = Vec::new();
        for index in self.active.active_indices() {
            let label = regions.map_or("UNMAPPED", |r| r.label(index));
            *active_by_region.entry(label.to_owned()).or_insert(0) += 1;
            if sampled_neuron_ids.len() < 16 {
                sampled_neuron_ids.push(self.graph.ids[index]);
            }
        }
        Ok(ActivityReport {
            active_neurons: self.active.active_count(),
            newly_active: 0,
            silenced: 0,
            active_sources_visited: 0,
            routed_edges: 0,
            contact_mass: 0,
            active_by_region,
            sampled_neuron_ids,
        })
    }

    /// One synchronous step, using preserved contact counts as experimental
    /// positive signals. This does not infer neurotransmitter signs or delays.
    pub fn step(&mut self, regions: Option<&RegionLabels>) -> io::Result<ActivityReport> {
        if self.active.len() != self.graph.ids.len() {
            return Err(invalid("graph/state size mismatch"));
        }
        if regions.is_some_and(|r| r.graph_ids.as_slice() != self.graph.ids.as_slice()) {
            return Err(invalid("region label neuron IDs do not match this graph"));
        }
        let before = self.active.clone();
        let mut incoming = vec![0_u128; self.graph.ids.len()];
        let mut routed_edges = 0_u64;
        let mut contact_mass = 0_u128;
        let mut visited_sources = 0_u64;

        for source in before.active_indices() {
            visited_sources += 1;
            let mut transmit = |target: usize, contacts: u64| -> io::Result<()> {
                incoming[target] = gates::add(incoming[target], u128::from(contacts))?;
                contact_mass = gates::add(contact_mass, u128::from(contacts))?;
                routed_edges += 1;
                Ok(())
            };
            if let Some((offsets, edges)) = &self.selected {
                for &edge in &edges[offsets[source]..offsets[source + 1]] {
                    transmit(self.graph.targets[edge] as usize, self.graph.counts[edge])?;
                }
            } else {
                for edge in
                    self.graph.offsets[source] as usize..self.graph.offsets[source + 1] as usize
                {
                    transmit(self.graph.targets[edge] as usize, self.graph.counts[edge])?;
                }
            }
            if let Some(reverse) = &self.reverse {
                for &(target, contacts) in &reverse[source] {
                    transmit(target, contacts)?;
                }
            }
        }

        let mut next = BooleanState::new(self.graph.ids.len());
        for (index, &signal) in incoming.iter().enumerate() {
            if gates::ge(signal, self.threshold) {
                next.set(index, true)
                    .map_err(|_| invalid("invalid neuron index"))?;
            }
        }
        let mut newly_active = 0_u64;
        let mut silenced = 0_u64;
        for (&old, &new) in before.words().iter().zip(next.words()) {
            newly_active += u64::from((new & !old).count_ones());
            silenced += u64::from((old & !new).count_ones());
        }
        self.active = next;
        let mut report = self.snapshot(regions)?;
        report.newly_active = newly_active;
        report.silenced = silenced;
        report.routed_edges = routed_edges;
        report.contact_mass = contact_mass;
        report.active_sources_visited = visited_sources;
        Ok(report)
    }
}

/// One bounded, cue-specific path used as evidence for associative recall.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CueContribution {
    pub cue_id: u64,
    pub strongest_path: u64,
    pub hops: u8,
}

/// A candidate reached from multiple distinct cue neurons.
/// Scores are deterministic *topological* evidence, not learned probabilities.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecallHit {
    pub neuron_id: u64,
    pub region: String,
    pub distinct_cues: usize,
    pub score: u128,
    pub nearest_hops: u8,
    pub evidence: Vec<CueContribution>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecallReport {
    pub hits: Vec<RecallHit>,
    pub edges_examined: u64,
    pub source_expansions: u64,
}

impl<'a> CircuitExplorer<'a> {
    /// Deterministic associative recall on the selected virtual circuit.
    ///
    /// Each cue independently explores at most three hops. At each hop only
    /// the strongest bottleneck path to a target is retained; between hops
    /// the strongest cue-to-target path wins. Each cue contributes at most
    /// once to a target, so repeated walks never inflate cue support.
    ///
    /// The edge budget is shared across all cues/hops; exhaustion is an
    /// error, not a silently truncated result. This computes structural
    /// associations, *not* a trained or biologically observed memory.
    pub fn recall(
        &self,
        cues: &[u64],
        max_hops: u8,
        min_distinct_cues: usize,
        top_k: usize,
        max_examined_edges: u64,
        regions: Option<&RegionLabels>,
    ) -> io::Result<RecallReport> {
        if cues.is_empty()
            || cues.len() > 64
            || !(1..=3).contains(&max_hops)
            || min_distinct_cues == 0
            || min_distinct_cues > cues.len()
            || !(1..=256).contains(&top_k)
            || !(1..=1_000_000).contains(&max_examined_edges)
        {
            return Err(invalid("recall limits or cue count invalid"));
        }
        if regions.is_some_and(|r| r.graph_ids.as_slice() != self.graph.ids.as_slice()) {
            return Err(invalid("region label neuron IDs do not match this graph"));
        }
        let mut cue_indices = Vec::with_capacity(cues.len());
        for &id in cues {
            cue_indices.push(
                self.graph
                    .index(id)
                    .ok_or_else(|| invalid("unknown recall cue neuron"))?,
            );
        }
        cue_indices.sort_unstable();
        cue_indices.dedup();
        if cue_indices.len() != cues.len() {
            return Err(invalid("duplicate recall cue neuron"));
        }

        let mut examined = 0_u64;
        let mut expansions = 0_u64;
        let mut matches: BTreeMap<usize, Vec<CueContribution>> = BTreeMap::new();
        for &cue in &cue_indices {
            let mut frontier = BTreeMap::from([(cue, u64::MAX)]);
            let mut best_for_cue: BTreeMap<usize, (u64, u8)> = BTreeMap::new();
            for hop in 1..=max_hops {
                let mut next: BTreeMap<usize, u64> = BTreeMap::new();
                for (&source, &path_strength) in &frontier {
                    expansions += 1;
                    let mut visit = |target: usize, contacts: u64| -> io::Result<()> {
                        examined += 1;
                        if examined > max_examined_edges {
                            return Err(invalid("recall edge exploration budget exceeded"));
                        }
                        if contacts > 0 && target != cue {
                            let strength = path_strength.min(contacts);
                            if let Some(current) = next.get_mut(&target) {
                                *current = (*current).max(strength);
                            } else {
                                if next.len() >= 100_000 {
                                    return Err(invalid("recall frontier memory budget exceeded"));
                                }
                                next.insert(target, strength);
                            }
                        }
                        Ok(())
                    };
                    if let Some((offsets, edges)) = &self.selected {
                        for &edge in &edges[offsets[source]..offsets[source + 1]] {
                            visit(self.graph.targets[edge] as usize, self.graph.counts[edge])?;
                        }
                    } else {
                        for edge in self.graph.offsets[source] as usize
                            ..self.graph.offsets[source + 1] as usize
                        {
                            visit(self.graph.targets[edge] as usize, self.graph.counts[edge])?;
                        }
                    }
                    if let Some(reverse) = &self.reverse {
                        for &(target, contacts) in &reverse[source] {
                            visit(target, contacts)?;
                        }
                    }
                }
                 for (&target, &strength) in &next {
                    let entry = best_for_cue.entry(target).or_insert((strength, hop));
                    if strength > entry.0 || (strength == entry.0 && hop < entry.1) {
                        *entry = (strength, hop);
                    }
                }
                frontier = next;
                if frontier.is_empty() {
                    break;
                }
            }
            for (target, (strength, hops)) in best_for_cue {
                matches.entry(target).or_default().push(CueContribution {
                    cue_id: self.graph.ids[cue],
                    strongest_path: strength,
                    hops,
                });
            }
        }

        let mut hits = Vec::new();
        for (target, evidence) in matches {
            if evidence.len() < min_distinct_cues {
                continue;
            }
            let mut score = 0_u128;
            let mut nearest_hops = u8::MAX;
            for item in &evidence {
                score = gates::add(score, u128::from(item.strongest_path))?;
                nearest_hops = nearest_hops.min(item.hops);
            }
            hits.push(RecallHit {
                neuron_id: self.graph.ids[target],
                region: regions.map_or("UNMAPPED", |r| r.label(target)).to_owned(),
                distinct_cues: evidence.len(),
                score,
                nearest_hops,
                evidence,
            });
        }
        hits.sort_unstable_by(|a, b| {
            b.distinct_cues
                .cmp(&a.distinct_cues)
                .then_with(|| b.score.cmp(&a.score))
                .then_with(|| a.nearest_hops.cmp(&b.nearest_hops))
                .then_with(|| a.neuron_id.cmp(&b.neuron_id))
        });
        hits.truncate(top_k);
        Ok(RecallReport {
            hits,
            edges_examined: examined,
            source_expansions: expansions,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn graph() -> Connectome {
        Connectome {
            ids: vec![10, 20, 30, 40],
            offsets: vec![0, 2, 3, 3, 3],
            targets: vec![1, 2, 3],
            counts: vec![1, 3, 2],
        }
    }

    #[test]
    fn invalid_public_state_never_indexes_graph_offsets() {
        let graph = graph();
        let mut explorer = CircuitExplorer::new(&graph, Routing::Directed, 1).unwrap();
        explorer.active = BooleanState::new(graph.ids.len() + 1);
        explorer.active.set(graph.ids.len(), true).unwrap();
        assert!(explorer.snapshot(None).is_err());
        assert!(explorer.step(None).is_err());
        assert!(explorer.stimulate(10).is_err());
    }

    #[test]
    fn region_labels_require_original_neuron_id_alignment() {
        let g = graph();
        let labels =
            RegionLabels::from_reader(&g, Cursor::new(b"neuron_id,region\n10,input\n")).unwrap();
        let swapped = Connectome {
            ids: vec![11, 20, 30, 40],
            offsets: g.offsets.clone(),
            targets: g.targets.clone(),
            counts: g.counts.clone(),
        };
        let mut explorer = CircuitExplorer::new(&swapped, Routing::Directed, 1).unwrap();
        explorer.stimulate(11).unwrap();
        assert!(explorer.snapshot(Some(&labels)).is_err());
        assert!(explorer.step(Some(&labels)).is_err());
        assert!(explorer.recall(&[11], 1, 1, 2, 100, Some(&labels)).is_err());
    }

    #[test]
    fn different_wiring_produces_distinct_activity_without_mutating_anatomy() {
        let g = graph();
        let original = g.clone();
        let mut all = CircuitExplorer::new(&g, Routing::Directed, 1).unwrap();
        all.stimulate(10).unwrap();
        let a = all.step(None).unwrap();
        assert_eq!(a.sampled_neuron_ids, vec![20, 30]);
        assert_eq!(a.routed_edges, 2);

        let mut selected = CircuitExplorer::new(&g, Routing::StrongestK(1), 1).unwrap();
        selected.stimulate(10).unwrap();
        let b = selected.step(None).unwrap();
        assert_eq!(b.sampled_neuron_ids, vec![30]);
        assert_eq!(b.routed_edges, 1);

        let mut mirrored = CircuitExplorer::new(&g, Routing::SymmetricHypothesis, 3).unwrap();
        mirrored.stimulate(30).unwrap();
        let c = mirrored.step(None).unwrap();
        assert_eq!(c.sampled_neuron_ids, vec![10]);
        assert_eq!(c.contact_mass, 3);
        assert_eq!(g, original);
    }

    #[test]
    fn routes_over_multiple_steps_and_reports_regions() {
        let g = graph();
        let csv = Cursor::new(b"neuron_id,region\n10,input\n20,relay\n40,output\n");
        let regions = RegionLabels::from_reader(&g, csv).unwrap();
        let mut explorer = CircuitExplorer::new(&g, Routing::Directed, 1).unwrap();
        explorer.stimulate(10).unwrap();
        assert_eq!(
            explorer.snapshot(Some(&regions)).unwrap().active_by_region["input"],
            1
        );
        let first = explorer.step(Some(&regions)).unwrap();
        assert_eq!(first.active_by_region["relay"], 1);
        assert_eq!(first.active_by_region["UNMAPPED"], 1);
        assert_eq!(first.newly_active, 2);
        assert_eq!(first.silenced, 1);
        let second = explorer.step(Some(&regions)).unwrap();
        assert_eq!(second.sampled_neuron_ids, vec![40]);
        assert_eq!(second.active_by_region["output"], 1);
    }

    #[test]
    fn recall_combines_distinct_cue_paths_and_retrieves_two_hop_associations() {
        let g = Connectome {
            ids: vec![10, 20, 30, 40, 50],
            offsets: vec![0, 2, 4, 5, 5, 5],
            targets: vec![2, 3, 2, 4, 4],
            counts: vec![2, 8, 5, 1, 3],
        };
        let regions =
            RegionLabels::from_reader(&g, Cursor::new(b"neuron_id,region\n30,association\n"))
                .unwrap();
        let directed = CircuitExplorer::new(&g, Routing::Directed, 1).unwrap();
        let before = directed.active.clone();
        let one = directed
            .recall(&[10, 20], 1, 2, 10, 100, Some(&regions))
            .unwrap();
        assert_eq!(one.hits.len(), 1);
        assert_eq!(one.hits[0].neuron_id, 30);
        assert_eq!(one.hits[0].region, "association");
        assert_eq!(one.hits[0].score, 7);
        assert_eq!(one.hits[0].distinct_cues, 2);
        assert_eq!(one.hits[0].evidence.len(), 2);

        let two = directed.recall(&[10, 20], 2, 2, 10, 100, None).unwrap();
        assert_eq!(
            two.hits.iter().map(|h| h.neuron_id).collect::<Vec<_>>(),
            vec![30, 50]
        );
        assert_eq!(two.hits[1].score, 5);
        assert_eq!(two.hits[1].nearest_hops, 2);
        assert_eq!(before, directed.active);
        assert_eq!(g.ids, vec![10, 20, 30, 40, 50]);

        let strongest = CircuitExplorer::new(&g, Routing::StrongestK(1), 1).unwrap();
        assert!(strongest
            .recall(&[10, 20], 1, 2, 10, 100, None)
            .unwrap()
            .hits
            .is_empty());
    }

    #[test]
    fn recall_fails_closed_on_budget_unknown_or_duplicate_cues() {
        let g = graph();
        let explorer = CircuitExplorer::new(&g, Routing::Directed, 1).unwrap();
        assert!(explorer.recall(&[10, 20], 2, 1, 10, 1, None).is_err());
        assert!(explorer.recall(&[10, 10], 1, 1, 10, 100, None).is_err());
        assert!(explorer.recall(&[10, 99], 1, 1, 10, 100, None).is_err());
        assert!(explorer.recall(&[10], 0, 1, 10, 100, None).is_err());
        assert!(explorer.recall(&[10], 1, 2, 10, 100, None).is_err());
    }

    #[test]
    fn refuses_invalid_region_metadata_and_budget() {
        let g = graph();
        assert!(
            RegionLabels::from_reader(&g, Cursor::new(b"neuron_id,region\n10,a\n10,b\n")).is_err()
        );
        assert!(
            RegionLabels::from_reader(&g, Cursor::new(b"neuron_id,region\n99,unknown\n")).is_err()
        );
        assert!(CircuitExplorer::new(&g, Routing::StrongestK(0), 1).is_err());
        assert!(CircuitExplorer::new(&g, Routing::Directed, 0).is_err());
    }
}
