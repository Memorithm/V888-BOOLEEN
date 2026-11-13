//! Explicit, reproducible recurrent wiring for large synthetic five-circle networks.
//!
//! Each node has exactly five data sources and five contextual sources (ten
//! directed contacts). All nodes update from one immutable, bit-packed
//! snapshot. No anatomical edges or neural physiology are inferred.
use crate::fourier5_dynamic::{FiveCircleLayer, GatedFiveCircle, ModulationError, WiredNeuron};
use crate::BooleanState;

pub const MIN_NETWORK_NODES: usize = 64;
pub const MAX_NETWORK_NODES: usize = 250_000;
pub const CONTACTS_PER_NODE: usize = 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SyntheticTopology {
    LocalRing,
    SeededPseudorandom,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NetworkConfig {
    pub nodes: usize,
    pub seed: u64,
    pub topology: SyntheticTopology,
    /// A value from 0..=256; each state bit is set when a seeded byte
    /// is lower than this value. This is a synthetic stimulus density.
    pub initial_activity_per_256: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecurrentError {
    InvalidSize,
    InvalidDensity,
    InvalidContact,
    MismatchedNetwork,
    WiringBudget,
    Model(ModulationError),
}

impl From<ModulationError> for RecurrentError {
    fn from(error: ModulationError) -> Self {
        Self::Model(error)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TickStats {
    pub tick: usize,
    pub active: u64,
    pub rising: u64,
    pub falling: u64,
}

#[derive(Clone, Debug)]
pub struct RecurrentFiveCircle {
    layer: FiveCircleLayer,
    state: BooleanState,
    ticks: usize,
    wiring_fingerprint: u64,
}

#[derive(Clone, Copy, Debug)]
struct Xorshift64(u64);

impl Xorshift64 {
    fn seeded(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9e37_79b9_7f4a_7c15
        } else {
            seed
        })
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// Noncryptographic, stable fingerprint of the explicit wiring.
/// This is NOT a security checksum and does not identify biological tissue.
fn fingerprint(mix: u64, index: usize) -> u64 {
    (mix ^ index as u64).wrapping_mul(0x0000_0100_0000_01b3)
}

fn distinct_sources(
    node: usize,
    count: usize,
    topology: SyntheticTopology,
    rng: &mut Xorshift64,
) -> Result<[usize; CONTACTS_PER_NODE], RecurrentError> {
    const RING_OFFSETS: [usize; CONTACTS_PER_NODE] = [1, 3, 7, 15, 31, 2, 5, 11, 23, 47];
    let mut sources = [0_usize; CONTACTS_PER_NODE];
    for index in 0..CONTACTS_PER_NODE {
        let selected = match topology {
            SyntheticTopology::LocalRing => (node + RING_OFFSETS[index]) % count,
            SyntheticTopology::SeededPseudorandom => {
                let mut attempts = 0;
                loop {
                    let candidate = (rng.next() % count as u64) as usize;
                    attempts += 1;
                    if candidate != node && !sources[..index].contains(&candidate) {
                        break candidate;
                    }
                    if attempts >= 256 {
                        return Err(RecurrentError::WiringBudget);
                    }
                }
            }
        };
        sources[index] = selected;
    }
    Ok(sources)
}

impl RecurrentFiveCircle {
    /// Construct a synthetic recurrent network, bounded to 250k neurons.
    /// A single previously trained GatedFiveCircle is shared by value;
    /// this does NOT train each node separately.
    pub fn seeded(config: NetworkConfig, model: GatedFiveCircle) -> Result<Self, RecurrentError> {
        if !(MIN_NETWORK_NODES..=MAX_NETWORK_NODES).contains(&config.nodes) {
            return Err(RecurrentError::InvalidSize);
        }
        if config.initial_activity_per_256 > 256 {
            return Err(RecurrentError::InvalidDensity);
        }
        let mut rng = Xorshift64::seeded(config.seed);
        let mut nodes = Vec::with_capacity(config.nodes);
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for node in 0..config.nodes {
            let sources = distinct_sources(node, config.nodes, config.topology, &mut rng)?;
            for &source in &sources {
                hash = fingerprint(hash, source);
            }
            let inputs = std::array::from_fn(|index| sources[index]);
            let context = std::array::from_fn(|index| sources[index + 5]);
            nodes.push(WiredNeuron {
                inputs,
                context,
                model,
            });
        }
        let layer = FiveCircleLayer::new(config.nodes, nodes)?;
        let mut state = BooleanState::new(config.nodes);
        for index in 0..config.nodes {
            let initial = (rng.next() & 255) < u64::from(config.initial_activity_per_256);
            if initial {
                state
                    .set(index, true)
                    .map_err(|_| RecurrentError::InvalidContact)?;
            }
        }
        Ok(Self {
            layer,
            state,
            ticks: 0,
            wiring_fingerprint: hash,
        })
    }

    #[must_use]
    pub fn node_count(&self) -> usize {
        self.state.len()
    }

    #[must_use]
    pub fn contact_count(&self) -> usize {
        self.state.len() * CONTACTS_PER_NODE
    }

    #[must_use]
    pub fn wiring_fingerprint(&self) -> u64 {
        self.wiring_fingerprint
    }

    #[must_use]
    pub fn state(&self) -> &BooleanState {
        &self.state
    }

    #[must_use]
    pub fn tick_count(&self) -> usize {
        self.ticks
    }

    /// Logical storage for the stored neuron records and one state snapshot;
    /// excludes Vec headers/allocator metadata, stack and temporary storage.
    #[must_use]
    pub fn estimated_state_and_wiring_bytes(&self) -> usize {
        self.node_count() * std::mem::size_of::<WiredNeuron>()
            + std::mem::size_of_val(self.state.words())
    }

    /// One synchronous recurrent update; compares original previous and next
    /// activity, not a stimulated/clamped snapshot.
    pub fn step(&mut self) -> Result<TickStats, RecurrentError> {
        let next = self.layer.step(&self.state)?;
        let mut rising = 0_u64;
        let mut falling = 0_u64;
        for (&previous, &new) in self.state.words().iter().zip(next.words()) {
            rising += u64::from((!previous & new).count_ones());
            falling += u64::from((previous & !new).count_ones());
        }
        let active = next.active_count();
        self.state = next;
        self.ticks += 1;
        Ok(TickStats {
            tick: self.ticks,
            active,
            rising,
            falling,
        })
    }

    /// Flip one state bit as a controlled perturbation for stability tests.
    /// The wiring and learned model are not modified.
    pub fn flip(&mut self, index: usize) -> Result<(), RecurrentError> {
        let value = self
            .state
            .get(index)
            .map_err(|_| RecurrentError::InvalidContact)?;
        self.state
            .set(index, !value)
            .map_err(|_| RecurrentError::InvalidContact)
    }

    /// Hamming distance between two network states. The caller must ensure
    /// the same topology (node count AND wiring fingerprint).
    pub fn hamming_distance(&self, other: &Self) -> Result<u64, RecurrentError> {
        if self.node_count() != other.node_count()
            || self.wiring_fingerprint != other.wiring_fingerprint
        {
            return Err(RecurrentError::MismatchedNetwork);
        }
        Ok(self
            .state
            .words()
            .iter()
            .zip(other.state.words())
            .map(|(&left, &right)| u64::from((left ^ right).count_ones()))
            .sum())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fourier5::{Circle, FiveCircle, CIRCLES};
    use crate::fourier5_dynamic::RadiusGate;

    fn model() -> GatedFiveCircle {
        let mut circles = [Circle::default(); CIRCLES];
        circles[1] = Circle {
            mask: 31,
            radius: 2,
            phase_pi: false,
        };
        let base = FiveCircle::new(circles, -1).unwrap();
        let mut gates = [RadiusGate::default(); CIRCLES];
        gates[0] = RadiusGate {
            context_mask: 5,
            boost: 3,
        };
        GatedFiveCircle::new(base, gates).unwrap()
    }

    fn config(topology: SyntheticTopology) -> NetworkConfig {
        NetworkConfig {
            nodes: 257,
            seed: 0x1234_5678,
            topology,
            initial_activity_per_256: 128,
        }
    }

    #[test]
    fn seeded_random_and_ring_are_reproducible_and_distinct() {
        let mut random =
            RecurrentFiveCircle::seeded(config(SyntheticTopology::SeededPseudorandom), model())
                .unwrap();
        let mut clone =
            RecurrentFiveCircle::seeded(config(SyntheticTopology::SeededPseudorandom), model())
                .unwrap();
        let ring =
            RecurrentFiveCircle::seeded(config(SyntheticTopology::LocalRing), model()).unwrap();
        assert_eq!(random.contact_count(), 2_570);
        assert_eq!(random.wiring_fingerprint(), clone.wiring_fingerprint());
        assert_ne!(random.wiring_fingerprint(), ring.wiring_fingerprint());
        assert_eq!(random.state(), clone.state());
        for tick in 1..=16 {
            let a = random.step().unwrap();
            let b = clone.step().unwrap();
            assert_eq!(a, b);
            assert_eq!(a.tick, tick);
            assert_eq!(random.state(), clone.state());
            assert_eq!(a.active + a.falling, b.active + b.falling);
        }
    }

    #[test]
    fn one_bit_perturbation_has_exact_hamming_accounting() {
        let mut original =
            RecurrentFiveCircle::seeded(config(SyntheticTopology::LocalRing), model()).unwrap();
        let mut perturbed = original.clone();
        assert_eq!(original.hamming_distance(&perturbed), Ok(0));
        perturbed.flip(64).unwrap();
        assert_eq!(original.hamming_distance(&perturbed), Ok(1));
        for _ in 0..8 {
            let stats = original.step().unwrap();
            let before = stats.active as i64 - stats.rising as i64 + stats.falling as i64;
            assert!(before >= 0);
            assert_eq!(stats.active, original.state().active_count());
            perturbed.step().unwrap();
            assert!(original.hamming_distance(&perturbed).unwrap() <= 257);
        }
        assert_eq!(original.tick_count(), 8);
    }

    #[test]
    fn guards_reject_out_of_range_sizes_density_and_contacts() {
        let invalid_size = NetworkConfig {
            nodes: 63,
            ..config(SyntheticTopology::LocalRing)
        };
        assert_eq!(
            RecurrentFiveCircle::seeded(invalid_size, model()).unwrap_err(),
            RecurrentError::InvalidSize
        );
        let invalid_density = NetworkConfig {
            initial_activity_per_256: 257,
            ..config(SyntheticTopology::LocalRing)
        };
        assert_eq!(
            RecurrentFiveCircle::seeded(invalid_density, model()).unwrap_err(),
            RecurrentError::InvalidDensity
        );
        let mut run =
            RecurrentFiveCircle::seeded(config(SyntheticTopology::LocalRing), model()).unwrap();
        assert_eq!(run.flip(257), Err(RecurrentError::InvalidContact));
        let other = RecurrentFiveCircle::seeded(
            NetworkConfig {
                nodes: 259,
                ..config(SyntheticTopology::LocalRing)
            },
            model(),
        )
        .unwrap();
        assert_eq!(
            run.hamming_distance(&other),
            Err(RecurrentError::MismatchedNetwork)
        );
    }

    #[test]
    fn storage_is_linear_and_state_is_bit_packed() {
        let network =
            RecurrentFiveCircle::seeded(config(SyntheticTopology::LocalRing), model()).unwrap();
        assert_eq!(network.state().words().len(), 5);
        assert_eq!(
            network.estimated_state_and_wiring_bytes(),
            257 * std::mem::size_of::<WiredNeuron>() + 5 * 8
        );
        assert_eq!(network.contact_count(), 257 * CONTACTS_PER_NODE);
    }

    #[test]
    fn no_spontaneous_dynamics_from_all_zero_state() {
        let mut network = RecurrentFiveCircle::seeded(
            NetworkConfig {
                initial_activity_per_256: 0,
                ..config(SyntheticTopology::SeededPseudorandom)
            },
            model(),
        )
        .unwrap();
        assert_eq!(network.state().active_count(), 0);
        for _ in 0..4 {
            assert_eq!(network.step().unwrap().active, 0);
        }
    }
}
