//! A recurrent graph of Boolean neurons. Dynamic rules are EXPERIMENTAL.
use crate::connectome::Connectome;
use crate::gates;
use crate::BooleanState;
use std::io::{self, ErrorKind};
fn invalid(message: &str) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, message)
}
pub struct BooleanEngine {
    pub active: BooleanState,
    threshold: u64,
}
impl BooleanEngine {
    pub fn new(graph: &Connectome, threshold: u64) -> io::Result<Self> {
        if threshold == 0 {
            return Err(invalid("threshold must be >=1"));
        }
        graph.validate()?;
        Ok(Self {
            active: BooleanState::new(graph.ids.len()),
            threshold,
        })
    }
    pub fn stimulate(&mut self, graph: &Connectome, id: u64) -> io::Result<()> {
        if self.active.len() != graph.ids.len() {
            return Err(invalid("graph/state size mismatch"));
        }
        let ix = graph.index(id).ok_or_else(|| invalid("unknown neuron"))?;
        self.active
            .set(ix, true)
            .map_err(|_| invalid("index error"))
    }
    /// Every active source sends its exact contact count to each destination;
    /// source and destination updates are synchronous, with no spontaneous activity.
    /// Neither biological sign nor delay nor plasticity is inferred.
    pub fn step(&mut self, graph: &Connectome) -> io::Result<u64> {
        if self.active.len() != graph.ids.len() {
            return Err(invalid("graph/state size mismatch"));
        }
        let mut incoming = vec![0_u128; graph.ids.len()];
        for source in 0..graph.ids.len() {
            if !self
                .active
                .get(source)
                .map_err(|_| invalid("index error"))?
            {
                continue;
            }
            for edge in graph.offsets[source]..graph.offsets[source + 1] {
                let edge = edge as usize;
                let target = graph.targets[edge] as usize;
                incoming[target] = gates::add(incoming[target], graph.counts[edge] as u128)?;
            }
        }
        let mut next = BooleanState::new(graph.ids.len());
        for (ix, &value) in incoming.iter().enumerate() {
            if gates::ge(value, self.threshold as u128) {
                next.set(ix, true).map_err(|_| invalid("index error"))?;
            }
        }
        self.active = next;
        Ok(self.active.active_count())
    }
}
