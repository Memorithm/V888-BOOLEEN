//! Frozen, deterministic composition of five-input Boolean Fourier neurons.
//!
//! Five-way layers form a tree with 5, 25, 125 or 625 input bits.
//! Explicit leaf routing permits matched wiring comparisons; there are
//! no invented BANC anatomical edges, implicit plasticity, or floating point.
//! A 32-entry compiled Boolean truth table is an equivalent *representation*
//! of each existing five-circle neuron, not a new neural learning mechanism.
use crate::fourier5::{Circle, FiveCircle, CIRCLES, TABLE_SIZE};
use crate::BooleanState;

pub const MAX_HIERARCHY_DEPTH: usize = 4;
pub const MAX_HIERARCHY_INPUTS: usize = 625;
pub const BITSLICE_LANES: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HierarchyError {
    InvalidDepth,
    InvalidRouting,
    InvalidLayerShape,
    InvalidInput,
    InvalidBatch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateKind {
    Parity,
    Or,
    And,
    Majority,
}

impl GateKind {
    /// Independent Boolean oracle for one five-input gate.
    #[must_use]
    pub fn oracle(self, input: u8) -> bool {
        let active = (input & 31).count_ones();
        match self {
            Self::Parity => active % 2 != 0,
            Self::Or => active != 0,
            Self::And => active == CIRCLES as u32,
            Self::Majority => active >= 3,
        }
    }

    /// Construct the same gate using only five ordinary Walsh-Fourier
    /// circles and an integer bias, not an external lookup table.
    #[must_use]
    pub fn five_circle(self) -> FiveCircle {
        let mut circles = [Circle::default(); CIRCLES];
        let bias = match self {
            Self::Parity => {
                circles[0] = Circle {
                    mask: 31,
                    radius: 1,
                    phase_pi: false,
                };
                0
            }
            Self::Or | Self::And | Self::Majority => {
                for (index, circle) in circles.iter_mut().enumerate() {
                    *circle = Circle {
                        mask: 1_u8 << index,
                        radius: 1,
                        phase_pi: false,
                    };
                }
                match self {
                    Self::Or => -4,
                    Self::And => 4,
                    Self::Majority => 0,
                    Self::Parity => unreachable!(),
                }
            }
        };
        FiveCircle::new(circles, bias).expect("fixed gate masks are valid")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeafRouting {
    Contiguous,
    Interleaved,
    SeededShuffle(u64),
}

#[derive(Clone, Debug)]
pub struct FiveCircleHierarchy {
    depth: usize,
    routing: Vec<u16>,
    layers: Vec<Vec<FiveCircle>>,
    truth_tables: Vec<Vec<u32>>,
}

fn next_random(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn routed_indices(count: usize, topology: LeafRouting) -> Vec<u16> {
    let mut routing: Vec<u16> = (0..count).map(|i| i as u16).collect();
    match topology {
        LeafRouting::Contiguous => {}
        LeafRouting::Interleaved => {
            let stride = count / CIRCLES;
            for (position, routed) in routing.iter_mut().enumerate() {
                *routed = ((position % CIRCLES) * stride + position / CIRCLES) as u16;
            }
        }
        LeafRouting::SeededShuffle(seed) => {
            let mut state = if seed == 0 {
                0x9e37_79b9_7f4a_7c15
            } else {
                seed
            };
            for index in (1..count).rev() {
                let position = (next_random(&mut state) % (index + 1) as u64) as usize;
                routing.swap(index, position);
            }
        }
    }
    routing
}

fn compiled_truth_table(model: FiveCircle) -> u32 {
    let mut truth = 0_u32;
    for input in 0..TABLE_SIZE {
        if model.predict(input as u8) == Ok(true) {
            truth |= 1_u32 << input;
        }
    }
    truth
}

fn gather_five(values: &[bool], first: usize) -> u8 {
    let mut word = 0_u8;
    for bit in 0..CIRCLES {
        word |= u8::from(values[first + bit]) << bit;
    }
    word
}

/// Compile one five-variable truth table to 31 bitwise Shannon multiplexers.
/// Each operand is a bit-plane holding 64 independent evaluation lanes.
/// The truth table itself is independently compiled from FiveCircle.predict.
fn shannon_64(truth: u32, input: [u64; CIRCLES]) -> u64 {
    let mut terms = [0_u64; TABLE_SIZE];
    for (row, term) in terms.iter_mut().enumerate() {
        *term = if truth & (1_u32 << row) != 0 {
            u64::MAX
        } else {
            0
        };
    }
    let mut size = TABLE_SIZE;
    for &selector in &input {
        for row in 0..size / 2 {
            terms[row] = (terms[2 * row] & !selector) | (terms[2 * row + 1] & selector);
        }
        size /= 2;
    }
    terms[0]
}

impl FiveCircleHierarchy {
    /// Construct an explicit, frozen tree. The routing must be a bijection
    /// over all leaf positions, and every layer must have exactly width/5
    /// neurons. No hidden overlapping or synthetic recurrent contacts.
    pub fn new(routing: Vec<u16>, layers: Vec<Vec<FiveCircle>>) -> Result<Self, HierarchyError> {
        let depth = layers.len();
        if depth == 0 || depth > MAX_HIERARCHY_DEPTH {
            return Err(HierarchyError::InvalidDepth);
        }
        let inputs = CIRCLES.pow(depth as u32);
        if routing.len() != inputs {
            return Err(HierarchyError::InvalidRouting);
        }
        let mut seen = vec![false; inputs];
        for &source in &routing {
            let index = usize::from(source);
            if index >= inputs || seen[index] {
                return Err(HierarchyError::InvalidRouting);
            }
            seen[index] = true;
        }
        let mut layer_input_width = inputs;
        for layer in &layers {
            layer_input_width /= CIRCLES;
            if layer.len() != layer_input_width {
                return Err(HierarchyError::InvalidLayerShape);
            }
        }
        if layer_input_width != 1 {
            return Err(HierarchyError::InvalidLayerShape);
        }
        let truth_tables = layers
            .iter()
            .map(|layer| layer.iter().copied().map(compiled_truth_table).collect())
            .collect();
        Ok(Self {
            depth,
            routing,
            layers,
            truth_tables,
        })
    }

    /// One homogeneous five-circle gate family per depth, with a reproducible
    /// permutation of all leaves. Gate modes are frozen, NOT trained.
    pub fn from_modes(modes: &[GateKind], topology: LeafRouting) -> Result<Self, HierarchyError> {
        if modes.is_empty() || modes.len() > MAX_HIERARCHY_DEPTH {
            return Err(HierarchyError::InvalidDepth);
        }
        let count = CIRCLES.pow(modes.len() as u32);
        let mut width = count;
        let mut layers = Vec::with_capacity(modes.len());
        for &mode in modes {
            width /= CIRCLES;
            layers.push(vec![mode.five_circle(); width]);
        }
        Self::new(routed_indices(count, topology), layers)
    }

    #[must_use]
    pub fn input_bits(&self) -> usize {
        self.routing.len()
    }

    #[must_use]
    pub fn depth(&self) -> usize {
        self.depth
    }

    #[must_use]
    pub fn neuron_count(&self) -> usize {
        self.layers.iter().map(Vec::len).sum()
    }

    #[must_use]
    pub fn directed_contacts(&self) -> usize {
        self.neuron_count() * CIRCLES
    }

    #[must_use]
    pub fn routing(&self) -> &[u16] {
        &self.routing
    }

    /// Produce another frozen hierarchy with the *same* Fourier neurons
    /// and compiled tables, but an explicitly validated leaf permutation.
    /// Reusing the compiled tables avoids recompiling 32 truth-table entries
    /// per neuron during bounded routing selection. No weights are changed.
    pub fn rerouted(&self, routing: Vec<u16>) -> Result<Self, HierarchyError> {
        if routing.len() != self.routing.len() {
            return Err(HierarchyError::InvalidRouting);
        }
        let mut seen = vec![false; self.routing.len()];
        for &source in &routing {
            let index = usize::from(source);
            if index >= seen.len() || seen[index] {
                return Err(HierarchyError::InvalidRouting);
            }
            seen[index] = true;
        }
        let mut updated = self.clone();
        updated.routing = routing;
        Ok(updated)
    }

    /// Frozen source routing is unchanged; only the five Walsh gains,
    /// phase bits and threshold of the FINAL neuron may be replaced.
    /// The equivalent Boolean truth table is recompiled exactly once.
    #[must_use]
    pub fn with_output_model(&self, model: FiveCircle) -> Self {
        let mut updated = self.clone();
        *updated
            .layers
            .last_mut()
            .expect("nonempty hierarchy")
            .first_mut()
            .expect("one output neuron") = model;
        *updated
            .truth_tables
            .last_mut()
            .expect("nonempty hierarchy")
            .first_mut()
            .expect("one output truth table") = compiled_truth_table(model);
        updated
    }

    #[must_use]
    pub fn output_model(&self) -> FiveCircle {
        self.layers.last().expect("nonempty hierarchy")[0]
    }

    /// Five presynaptic bits observed at the final neuron's inputs.
    /// Used to fit ONLY that readout's parameters, never intermediate gates.
    /// Labels and test sets have no influence on this observation.
    pub fn output_input(&self, state: &BooleanState) -> Result<u8, HierarchyError> {
        if state.len() != self.input_bits() {
            return Err(HierarchyError::InvalidInput);
        }
        let mut current: Vec<bool> = self
            .routing
            .iter()
            .map(|&source| state.get(usize::from(source)).expect("validated routing"))
            .collect();
        for layer in self.truth_tables.iter().take(self.depth - 1) {
            let mut next = Vec::with_capacity(layer.len());
            for (index, &truth) in layer.iter().enumerate() {
                let word = gather_five(&current, index * CIRCLES);
                next.push((truth >> word) & 1 != 0);
            }
            current = next;
        }
        Ok(gather_five(&current, 0))
    }

    #[must_use]
    pub fn logical_wiring_and_model_bytes(&self) -> usize {
        std::mem::size_of_val(self.routing.as_slice())
            + self.neuron_count() * std::mem::size_of::<FiveCircle>()
    }

    #[must_use]
    pub fn compiled_truth_bytes(&self) -> usize {
        self.neuron_count() * std::mem::size_of::<u32>()
    }

    /// Scalar Fourier-Walsh inference, one five-circle neuron at a time.
    pub fn infer_scalar(&self, state: &BooleanState) -> Result<bool, HierarchyError> {
        if state.len() != self.input_bits() {
            return Err(HierarchyError::InvalidInput);
        }
        let mut current: Vec<bool> = self
            .routing
            .iter()
            .map(|&index| state.get(usize::from(index)).expect("routing validated"))
            .collect();
        for layer in &self.layers {
            let mut next = Vec::with_capacity(layer.len());
            for (index, neuron) in layer.iter().enumerate() {
                let input = gather_five(&current, index * CIRCLES);
                next.push(neuron.predict(input).expect("five source bits"));
            }
            current = next;
        }
        Ok(current[0])
    }

    /// Scalar equivalent lookup inference, with the EXACT same routed tree.
    pub fn infer_lut_scalar(&self, state: &BooleanState) -> Result<bool, HierarchyError> {
        if state.len() != self.input_bits() {
            return Err(HierarchyError::InvalidInput);
        }
        let mut current: Vec<bool> = self
            .routing
            .iter()
            .map(|&index| state.get(usize::from(index)).expect("routing validated"))
            .collect();
        for layer in &self.truth_tables {
            let mut next = Vec::with_capacity(layer.len());
            for (index, &truth) in layer.iter().enumerate() {
                let input = gather_five(&current, index * CIRCLES);
                next.push((truth >> input) & 1 != 0);
            }
            current = next;
        }
        Ok(current[0])
    }

    /// Bit-sliced evaluation on 64 already transposed input patterns.
    /// The source bit-planes are a read-only view; every routing leaf
    /// is read once to assemble the first layer, regardless of depth.
    pub fn infer_bitplanes(&self, planes: &[u64]) -> Result<u64, HierarchyError> {
        if planes.len() != self.input_bits() {
            return Err(HierarchyError::InvalidInput);
        }
        let mut current: Vec<u64> = self
            .routing
            .iter()
            .map(|&index| planes[usize::from(index)])
            .collect();
        for layer in &self.truth_tables {
            let mut next = Vec::with_capacity(layer.len());
            for (index, &truth) in layer.iter().enumerate() {
                let inputs = std::array::from_fn(|bit| current[index * CIRCLES + bit]);
                next.push(shannon_64(truth, inputs));
            }
            current = next;
        }
        Ok(current[0])
    }

    /// Transpose exactly 64 immutable bit-packed BooleanState records
    /// into input-bit-major lanes, with no label-dependent preprocessing.
    pub fn transpose64(states: &[BooleanState]) -> Result<Vec<u64>, HierarchyError> {
        if states.len() != BITSLICE_LANES {
            return Err(HierarchyError::InvalidBatch);
        }
        let width = states[0].len();
        if width == 0 || width > MAX_HIERARCHY_INPUTS {
            return Err(HierarchyError::InvalidInput);
        }
        if states.iter().any(|state| state.len() != width) {
            return Err(HierarchyError::InvalidInput);
        }
        let mut planes = vec![0_u64; width];
        for (lane, state) in states.iter().enumerate() {
            for bit in state.active_indices() {
                planes[bit] |= 1_u64 << lane;
            }
        }
        Ok(planes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state_from_word(width: usize, seed: u64) -> BooleanState {
        let mut state = BooleanState::new(width);
        for bit in 0..width {
            if (seed.rotate_left((bit % 64) as u32)
                ^ (bit as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
                & 1
                != 0
            {
                state.set(bit, true).unwrap();
            }
        }
        state
    }

    fn reference(state: &BooleanState, route: &[u16], modes: &[GateKind]) -> bool {
        let mut current: Vec<bool> = route
            .iter()
            .map(|&src| state.get(usize::from(src)).unwrap())
            .collect();
        for &mode in modes {
            current = current
                .chunks_exact(CIRCLES)
                .map(|group| {
                    let mut bits = 0_u8;
                    for (bit, &value) in group.iter().enumerate() {
                        bits |= u8::from(value) << bit;
                    }
                    mode.oracle(bits)
                })
                .collect();
        }
        current[0]
    }

    #[test]
    fn five_circle_gates_match_independent_boolean_oracle_exhaustively() {
        for kind in [
            GateKind::Parity,
            GateKind::Or,
            GateKind::And,
            GateKind::Majority,
        ] {
            let model = kind.five_circle();
            let truth = compiled_truth_table(model);
            for input in 0..32 {
                let expected = kind.oracle(input);
                assert_eq!(model.predict(input), Ok(expected));
                assert_eq!(truth & (1_u32 << input) != 0, expected);
                let planes: [u64; CIRCLES] =
                    std::array::from_fn(|bit| if input & (1 << bit) != 0 { u64::MAX } else { 0 });
                assert_eq!(
                    shannon_64(truth, planes),
                    if expected { u64::MAX } else { 0 }
                );
            }
        }
    }

    #[test]
    fn routing_modes_are_bijections_and_depth_four_is_625_bits() {
        let modes = [GateKind::Parity; 4];
        for topology in [
            LeafRouting::Contiguous,
            LeafRouting::Interleaved,
            LeafRouting::SeededShuffle(17),
        ] {
            let tree = FiveCircleHierarchy::from_modes(&modes, topology).unwrap();
            let mut sorted = tree.routing().to_vec();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..625).map(|x| x as u16).collect::<Vec<_>>());
            assert_eq!(tree.input_bits(), 625);
            assert_eq!(tree.neuron_count(), 156);
            assert_eq!(tree.directed_contacts(), 780);
            let a = state_from_word(625, 0x35);
            let odd = a.active_count() % 2 != 0;
            assert_eq!(tree.infer_scalar(&a), Ok(odd));
            assert_eq!(tree.infer_lut_scalar(&a), Ok(odd));
        }
    }

    #[test]
    fn bit_sliced_64_lanes_match_scalar_and_independent_oracle() {
        for modes in [
            &[GateKind::Parity, GateKind::Parity][..],
            &[GateKind::Or, GateKind::And, GateKind::Majority][..],
            &[
                GateKind::Majority,
                GateKind::Parity,
                GateKind::And,
                GateKind::Or,
            ][..],
        ] {
            let tree =
                FiveCircleHierarchy::from_modes(modes, LeafRouting::SeededShuffle(123)).unwrap();
            let states: Vec<_> = (0..BITSLICE_LANES)
                .map(|seed| state_from_word(tree.input_bits(), 0x1732_91a5 ^ seed as u64))
                .collect();
            let planes = FiveCircleHierarchy::transpose64(&states).unwrap();
            let packed = tree.infer_bitplanes(&planes).unwrap();
            for (lane, state) in states.iter().enumerate() {
                let expected = reference(state, tree.routing(), modes);
                assert_eq!(tree.infer_scalar(state), Ok(expected));
                assert_eq!(tree.infer_lut_scalar(state), Ok(expected));
                assert_eq!(packed & (1_u64 << lane) != 0, expected);
            }
        }
    }

    #[test]
    fn rejects_invalid_shapes_duplicate_sources_and_batches() {
        assert!(matches!(
            FiveCircleHierarchy::from_modes(&[], LeafRouting::Contiguous),
            Err(HierarchyError::InvalidDepth)
        ));
        assert!(matches!(
            FiveCircleHierarchy::from_modes(&[GateKind::Parity; 5], LeafRouting::Contiguous),
            Err(HierarchyError::InvalidDepth)
        ));
        let gate = GateKind::Parity.five_circle();
        assert!(matches!(
            FiveCircleHierarchy::new(vec![0; 5], vec![vec![gate]]),
            Err(HierarchyError::InvalidRouting)
        ));
        assert!(matches!(
            FiveCircleHierarchy::new(vec![0, 1, 2, 3, 5], vec![vec![gate]]),
            Err(HierarchyError::InvalidRouting)
        ));
        assert!(matches!(
            FiveCircleHierarchy::new(vec![0, 1, 2, 3, 4], vec![vec![gate, gate]]),
            Err(HierarchyError::InvalidLayerShape)
        ));
        let tree =
            FiveCircleHierarchy::from_modes(&[GateKind::Parity], LeafRouting::Contiguous).unwrap();
        assert_eq!(
            tree.infer_scalar(&BooleanState::new(6)),
            Err(HierarchyError::InvalidInput)
        );
        assert_eq!(
            tree.infer_bitplanes(&[0; 4]),
            Err(HierarchyError::InvalidInput)
        );
        assert_eq!(
            FiveCircleHierarchy::transpose64(&vec![BooleanState::new(5); 63]),
            Err(HierarchyError::InvalidBatch)
        );
        let mut malformed = vec![BooleanState::new(5); 64];
        malformed[63] = BooleanState::new(6);
        assert_eq!(
            FiveCircleHierarchy::transpose64(&malformed),
            Err(HierarchyError::InvalidInput)
        );
    }

    #[test]
    fn scalar_and_compiled_lut_have_identical_output_for_every_five_bit_input() {
        for kind in [
            GateKind::Parity,
            GateKind::Or,
            GateKind::And,
            GateKind::Majority,
        ] {
            let tree = FiveCircleHierarchy::from_modes(&[kind], LeafRouting::Contiguous).unwrap();
            for x in 0..32 {
                let mut state = BooleanState::new(5);
                for bit in 0..5 {
                    state.set(bit, x & (1 << bit) != 0).unwrap();
                }
                assert_eq!(tree.infer_scalar(&state), Ok(kind.oracle(x)));
                assert_eq!(tree.infer_lut_scalar(&state), Ok(kind.oracle(x)));
            }
        }
    }
}
