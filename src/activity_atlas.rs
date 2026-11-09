//! Persistent per-neuron activity history on the original V888 connectome.
//! Experimental activation observations, not measured insect physiology.
//! Uses original root IDs and an exact graph fingerprint for safe inspection.
use crate::connectome::Connectome;
use crate::BooleanState;
use std::io::{self, Read, Write};

const MAGIC: &[u8; 8] = b"V888ATL1";
const NEVER: u32 = u32::MAX;
const MAX_TICKS: u32 = 1_000_000;

fn bad(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
fn put_u32<W: Write>(w: &mut W, n: u32) -> io::Result<()> {
    w.write_all(&n.to_le_bytes())
}
fn put_u64<W: Write>(w: &mut W, n: u64) -> io::Result<()> {
    w.write_all(&n.to_le_bytes())
}
fn get_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut bytes = [0_u8; 4];
    r.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}
fn get_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut bytes = [0_u8; 8];
    r.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

/// Fingerprint is an accidental-mismatch guard, not a cryptographic signature.
fn graph_fingerprint(graph: &Connectome) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for value in graph
        .ids
        .iter()
        .copied()
        .chain(graph.offsets.iter().copied())
        .chain(graph.targets.iter().map(|&target| u64::from(target)))
        .chain(graph.counts.iter().copied())
    {
        for byte in value.to_le_bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NeuronActivity {
    pub neuron_id: u64,
    pub active_ticks: u32,
    pub rising_edges: u32,
    pub falling_edges: u32,
    pub longest_active_streak: u32,
    pub first_active_tick: Option<u32>,
    pub last_active_tick: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TickSummary {
    pub tick: u32,
    pub active: u64,
    pub newly_active: u64,
    pub silenced: u64,
}

/// Per-neuron change across two equal-duration simulation experiments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivationDelta {
    pub neuron_id: u64,
    pub before_ticks: u32,
    pub after_ticks: u32,
    pub delta_ticks: i64,
}

#[derive(Debug, Clone)]
pub struct ActivityAtlas {
    graph_hash: u64,
    ticks: u32,
    active_ticks: Vec<u32>,
    rising: Vec<u32>,
    falling: Vec<u32>,
    longest_streak: Vec<u32>,
    ongoing_streak: Vec<u32>,
    first_seen: Vec<u32>,
    last_seen: Vec<u32>,
    previous: BooleanState,
    timeline: Vec<TickSummary>,
}

impl ActivityAtlas {
    pub fn new(graph: &Connectome) -> io::Result<Self> {
        graph.validate()?;
        let n = graph.ids.len();
        Ok(Self {
            graph_hash: graph_fingerprint(graph),
            ticks: 0,
            active_ticks: vec![0; n],
            rising: vec![0; n],
            falling: vec![0; n],
            longest_streak: vec![0; n],
            ongoing_streak: vec![0; n],
            first_seen: vec![NEVER; n],
            last_seen: vec![NEVER; n],
            previous: BooleanState::new(n),
            timeline: Vec::new(),
        })
    }

    fn validate_graph(&self, graph: &Connectome) -> io::Result<()> {
        if self.previous.len() != graph.ids.len() || self.graph_hash != graph_fingerprint(graph) {
            return Err(bad("activity log belongs to a different graph"));
        }
        Ok(())
    }

    pub fn ticks(&self) -> u32 {
        self.ticks
    }

    pub fn timeline(&self) -> &[TickSummary] {
        &self.timeline
    }

    /// Record a complete synchronous state, including tick zero if desired.
    /// Work scales with active neurons and transitions, not dense neurons per tick.
    pub fn observe(&mut self, graph: &Connectome, state: &BooleanState) -> io::Result<TickSummary> {
        // The caller must keep the graph fixed for a recording session.
        // Avoid hashing all 11M+ source edges at every simulated tick.
        if state.len() != graph.ids.len() || self.previous.len() != graph.ids.len() {
            return Err(bad("state and graph size mismatch"));
        }
        if self.ticks == MAX_TICKS {
            return Err(bad("activity tick limit exceeded"));
        }
        let tick = self.ticks;
        let mut active = 0_u64;
        let mut newly_active = 0_u64;
        let mut silenced = 0_u64;
        for i in state.active_indices() {
            active += 1;
            self.active_ticks[i] += 1;
            self.last_seen[i] = tick;
            let was_active = self.previous.get(i).map_err(|_| bad("state index error"))?;
            if was_active {
                self.ongoing_streak[i] += 1;
            } else {
                newly_active += 1;
                self.rising[i] += 1;
                self.ongoing_streak[i] = 1;
                if self.first_seen[i] == NEVER {
                    self.first_seen[i] = tick;
                }
            }
            self.longest_streak[i] = self.longest_streak[i].max(self.ongoing_streak[i]);
        }
        for i in self.previous.active_indices() {
            if !state.get(i).map_err(|_| bad("state index error"))? {
                silenced += 1;
                self.falling[i] += 1;
                self.ongoing_streak[i] = 0;
            }
        }
        self.previous = state.clone();
        self.ticks += 1;
        let summary = TickSummary {
            tick,
            active,
            newly_active,
            silenced,
        };
        self.timeline.push(summary);
        Ok(summary)
    }

    pub fn neuron(&self, graph: &Connectome, id: u64) -> io::Result<Option<NeuronActivity>> {
        self.validate_graph(graph)?;
        Ok(graph.index(id).map(|index| self.neuron_at(graph, index)))
    }

    fn neuron_at(&self, graph: &Connectome, i: usize) -> NeuronActivity {
        NeuronActivity {
            neuron_id: graph.ids[i],
            active_ticks: self.active_ticks[i],
            rising_edges: self.rising[i],
            falling_edges: self.falling[i],
            longest_active_streak: self.longest_streak[i],
            first_active_tick: (self.first_seen[i] != NEVER).then_some(self.first_seen[i]),
            last_active_tick: (self.last_seen[i] != NEVER).then_some(self.last_seen[i]),
        }
    }

    /// Rank by activation frequency, then first-seen tick, then original root ID.
    pub fn top_neurons(&self, graph: &Connectome, limit: usize) -> io::Result<Vec<NeuronActivity>> {
        self.validate_graph(graph)?;
        let mut indices: Vec<usize> = (0..graph.ids.len())
            .filter(|&i| self.active_ticks[i] > 0)
            .collect();
        indices.sort_unstable_by(|a, b| {
            self.active_ticks[*b]
                .cmp(&self.active_ticks[*a])
                .then_with(|| self.first_seen[*a].cmp(&self.first_seen[*b]))
                .then_with(|| graph.ids[*a].cmp(&graph.ids[*b]))
        });
        indices.truncate(limit.min(10_000));
        Ok(indices
            .into_iter()
            .map(|i| self.neuron_at(graph, i))
            .collect())
    }

    /// Compare two experiments on the same graph, in descending order of
    /// absolute activation-frequency difference. Requires identical durations
    /// to avoid comparing different observation budgets.
    ///
    /// Positive delta means more active ticks in `other`. Only changed neurons
    /// are returned. Neither learning nor biological causation is inferred.
    pub fn compare_activation(
        &self,
        other: &Self,
        graph: &Connectome,
        limit: usize,
    ) -> io::Result<Vec<ActivationDelta>> {
        self.validate_graph(graph)?;
        other.validate_graph(graph)?;
        if self.ticks != other.ticks {
            return Err(bad("activity comparison requires identical tick budgets"));
        }
        let mut indices: Vec<usize> = (0..graph.ids.len())
            .filter(|&i| self.active_ticks[i] != other.active_ticks[i])
            .collect();
        indices.sort_unstable_by(|a, b| {
            self.active_ticks[*b]
                .abs_diff(other.active_ticks[*b])
                .cmp(&self.active_ticks[*a].abs_diff(other.active_ticks[*a]))
                .then_with(|| graph.ids[*a].cmp(&graph.ids[*b]))
        });
        indices.truncate(limit.min(10_000));
        Ok(indices
            .into_iter()
            .map(|i| ActivationDelta {
                neuron_id: graph.ids[i],
                before_ticks: self.active_ticks[i],
                after_ticks: other.active_ticks[i],
                delta_ticks: i64::from(other.active_ticks[i]) - i64::from(self.active_ticks[i]),
            })
            .collect())
    }

    /// Lossless state/timeline archive; checked graph identity prevents
    /// accidentally interpreting observations against another connectome.
    pub fn write<W: Write>(&self, graph: &Connectome, mut w: W) -> io::Result<()> {
        self.validate_graph(graph)?;
        w.write_all(MAGIC)?;
        put_u64(&mut w, self.graph_hash)?;
        put_u64(&mut w, graph.ids.len() as u64)?;
        put_u32(&mut w, self.ticks)?;
        for array in [
            &self.active_ticks,
            &self.rising,
            &self.falling,
            &self.longest_streak,
            &self.ongoing_streak,
            &self.first_seen,
            &self.last_seen,
        ] {
            for &value in array {
                put_u32(&mut w, value)?;
            }
        }
        for &word in self.previous.words() {
            put_u64(&mut w, word)?;
        }
        for tick in &self.timeline {
            put_u64(&mut w, tick.active)?;
            put_u64(&mut w, tick.newly_active)?;
            put_u64(&mut w, tick.silenced)?;
        }
        Ok(())
    }

    pub fn read<R: Read>(graph: &Connectome, mut r: R) -> io::Result<Self> {
        graph.validate()?;
        let mut magic = [0_u8; 8];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(bad("unknown activity archive format"));
        }
        let hash = get_u64(&mut r)?;
        let n = get_u64(&mut r)?;
        let ticks = get_u32(&mut r)?;
        if hash != graph_fingerprint(graph) || n != graph.ids.len() as u64 {
            return Err(bad("activity archive graph mismatch"));
        }
        if ticks > MAX_TICKS {
            return Err(bad("activity tick limit exceeded"));
        }
        let mut arrays = Vec::with_capacity(7);
        for _ in 0..7 {
            let mut array = Vec::with_capacity(graph.ids.len());
            for _ in 0..graph.ids.len() {
                array.push(get_u32(&mut r)?);
            }
            arrays.push(array);
        }
        let mut previous = BooleanState::new(graph.ids.len());
        for word_index in 0..graph.ids.len().div_ceil(64) {
            let mut word = get_u64(&mut r)?;
            while word != 0 {
                let bit = word.trailing_zeros() as usize;
                let neuron = word_index * 64 + bit;
                if neuron >= graph.ids.len() {
                    return Err(bad("nonzero archive padding bits"));
                }
                previous
                    .set(neuron, true)
                    .map_err(|_| bad("archive index"))?;
                word &= word - 1;
            }
        }
        let mut timeline = Vec::with_capacity(ticks as usize);
        for tick in 0..ticks {
            let active = get_u64(&mut r)?;
            let newly_active = get_u64(&mut r)?;
            let silenced = get_u64(&mut r)?;
            if active > n || newly_active > active || silenced > n {
                return Err(bad("invalid activity tick counters"));
            }
            timeline.push(TickSummary {
                tick,
                active,
                newly_active,
                silenced,
            });
        }
        let mut extra = [0_u8; 1];
        if r.read(&mut extra)? != 0 {
            return Err(bad("trailing activity archive bytes"));
        }
        let mut iter = arrays.into_iter();
        let atlas = Self {
            graph_hash: hash,
            ticks,
            active_ticks: iter.next().ok_or_else(|| bad("missing activity array"))?,
            rising: iter.next().ok_or_else(|| bad("missing activity array"))?,
            falling: iter.next().ok_or_else(|| bad("missing activity array"))?,
            longest_streak: iter.next().ok_or_else(|| bad("missing activity array"))?,
            ongoing_streak: iter.next().ok_or_else(|| bad("missing activity array"))?,
            first_seen: iter.next().ok_or_else(|| bad("missing activity array"))?,
            last_seen: iter.next().ok_or_else(|| bad("missing activity array"))?,
            previous,
            timeline,
        };
        for i in 0..graph.ids.len() {
            if atlas.active_ticks[i] > ticks
                || atlas.rising[i] > atlas.active_ticks[i]
                || atlas.falling[i] > ticks
                || atlas.longest_streak[i] > atlas.active_ticks[i]
                || atlas.ongoing_streak[i] > atlas.longest_streak[i]
                || ((atlas.first_seen[i] == NEVER) != (atlas.active_ticks[i] == 0))
                || (atlas.first_seen[i] != NEVER && atlas.first_seen[i] >= ticks)
                || (atlas.last_seen[i] != NEVER && atlas.last_seen[i] >= ticks)
            {
                return Err(bad("invalid activity neuron counters"));
            }
        }
        if atlas
            .timeline
            .last()
            .is_some_and(|last| last.active != atlas.previous.active_count())
        {
            return Err(bad("last tick does not match archive bit state"));
        }
        Ok(atlas)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sample() -> Connectome {
        Connectome {
            ids: vec![10, 20, 30],
            offsets: vec![0, 1, 2, 2],
            targets: vec![1, 2],
            counts: vec![2, 3],
        }
    }
    #[test]
    fn records_transitions_and_reopens_exactly() {
        let graph = sample();
        let mut atlas = ActivityAtlas::new(&graph).unwrap();
        let mut state = BooleanState::new(3);
        state.set(0, true).unwrap();
        atlas.observe(&graph, &state).unwrap();
        state.set(1, true).unwrap();
        atlas.observe(&graph, &state).unwrap();
        state.set(0, false).unwrap();
        atlas.observe(&graph, &state).unwrap();
        assert_eq!(atlas.ticks(), 3);
        let first = atlas.neuron(&graph, 10).unwrap().unwrap();
        assert_eq!(first.active_ticks, 2);
        assert_eq!(first.rising_edges, 1);
        assert_eq!(first.falling_edges, 1);
        assert_eq!(first.longest_active_streak, 2);
        assert_eq!(first.first_active_tick, Some(0));
        assert_eq!(first.last_active_tick, Some(1));
        let top = atlas.top_neurons(&graph, 2).unwrap();
        assert_eq!(top[0].neuron_id, 10);
        assert_eq!(top[1].neuron_id, 20);
        let mut data = Vec::new();
        atlas.write(&graph, &mut data).unwrap();
        let restored = ActivityAtlas::read(&graph, data.as_slice()).unwrap();
        assert_eq!(restored.timeline(), atlas.timeline());
        assert_eq!(restored.neuron(&graph, 10).unwrap(), Some(first));
        assert_eq!(
            restored.top_neurons(&graph, 3).unwrap(),
            atlas.top_neurons(&graph, 3).unwrap()
        );
    }
    #[test]
    fn compares_equal_budget_experiments_and_rejects_unequal() {
        let graph = sample();
        let mut before = ActivityAtlas::new(&graph).unwrap();
        let mut after = ActivityAtlas::new(&graph).unwrap();
        let mut a = BooleanState::new(3);
        let mut b = BooleanState::new(3);
        a.set(0, true).unwrap();
        b.set(1, true).unwrap();
        before.observe(&graph, &a).unwrap();
        after.observe(&graph, &b).unwrap();
        let deltas = before.compare_activation(&after, &graph, 3).unwrap();
        assert_eq!(
            deltas,
            vec![
                ActivationDelta {
                    neuron_id: 10,
                    before_ticks: 1,
                    after_ticks: 0,
                    delta_ticks: -1
                },
                ActivationDelta {
                    neuron_id: 20,
                    before_ticks: 0,
                    after_ticks: 1,
                    delta_ticks: 1
                },
            ]
        );
        assert!(before
            .compare_activation(&after, &graph, 0)
            .unwrap()
            .is_empty());
        after.observe(&graph, &b).unwrap();
        assert!(before.compare_activation(&after, &graph, 3).is_err());
        let mut changed = graph.clone();
        changed.counts[0] += 1;
        assert!(before.compare_activation(&after, &changed, 3).is_err());
    }
    #[test]
    fn rejects_mismatch_and_truncation() {
        let graph = sample();
        let mut atlas = ActivityAtlas::new(&graph).unwrap();
        let mut state = BooleanState::new(3);
        state.set(0, true).unwrap();
        atlas.observe(&graph, &state).unwrap();
        let mut data = Vec::new();
        atlas.write(&graph, &mut data).unwrap();
        let mut changed = graph.clone();
        changed.counts[0] += 1;
        assert!(ActivityAtlas::read(&changed, data.as_slice()).is_err());
        data.pop();
        assert!(ActivityAtlas::read(&graph, data.as_slice()).is_err());
    }
}
