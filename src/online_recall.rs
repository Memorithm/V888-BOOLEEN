//! Streaming activation statistics for explicitly selected experimental subcircuits.
//! Observations are simulated states, never evidence of biological firing.
use crate::subcircuit_recall::{RecallError, Subcircuit};
use crate::BooleanState;

/// Bounded-memory online recall. No snapshots or anatomical edges are stored.
#[derive(Clone, Debug)]
pub struct OnlineRecall {
    circuit: Subcircuit,
    counts: Vec<u64>,
    ticks: u64,
    current_streak: u64,
    longest_streak: u64,
}

impl OnlineRecall {
    pub fn new(circuit: Subcircuit) -> Self {
        let counts = vec![0; circuit.members().len()];
        Self {
            circuit,
            counts,
            ticks: 0,
            current_streak: 0,
            longest_streak: 0,
        }
    }

    /// Update atomically: invalid dimensions or counter overflow leave all counters unchanged.
    /// A matching observation has at least `minimum_active` selected bits set.
    pub fn observe(
        &mut self,
        state: &BooleanState,
        minimum_active: usize,
    ) -> Result<bool, RecallError> {
        if minimum_active == 0 {
            return Err(RecallError::ZeroBudget);
        }
        // Validate before modifying counters.
        let matches = self.circuit.recall(state, minimum_active)?;
        let next_ticks = self.ticks.checked_add(1).ok_or(RecallError::ZeroBudget)?;
        let next_streak = if matches {
            self.current_streak
                .checked_add(1)
                .ok_or(RecallError::ZeroBudget)?
        } else {
            0
        };
        let mut increments = Vec::with_capacity(self.counts.len());
        for (position, &member) in self.circuit.members().iter().enumerate() {
            if state.get(member) == Ok(true) {
                self.counts[position]
                    .checked_add(1)
                    .ok_or(RecallError::ZeroBudget)?;
                increments.push(position);
            }
        }
        self.ticks = next_ticks;
        self.current_streak = next_streak;
        self.longest_streak = self.longest_streak.max(next_streak);
        for position in increments {
            self.counts[position] += 1;
        }
        Ok(matches)
    }

    /// Observe the current state of the actual BooleanEngine, without copying a snapshot.
    /// The caller controls when the engine is stimulated or stepped. This is
    /// experimental activation tracking, not a biological firing measurement.
    pub fn observe_engine_state(
        &mut self,
        engine: &crate::engine::BooleanEngine,
        minimum_active: usize,
    ) -> Result<bool, RecallError> {
        self.observe(&engine.active, minimum_active)
    }

    /// Advance a validated synthetic graph and observe each resulting engine state.
    /// The initial state is not observed. At most 1000 steps are permitted, matching
    /// the CLI engine bound. A step failure preserves previously completed steps;
    /// no rollback of engine dynamics is implied. No anatomical edges are created.
    pub fn observe_engine_steps(
        &mut self,
        engine: &mut crate::engine::BooleanEngine,
        graph: &crate::connectome::Connectome,
        minimum_active: usize,
        steps: usize,
    ) -> std::io::Result<Vec<bool>> {
        use std::io::{Error, ErrorKind};
        if steps == 0 || steps > 1000 || minimum_active == 0 {
            return Err(Error::new(ErrorKind::InvalidInput, "invalid recall budget"));
        }
        graph.validate()?;
        if engine.active.len() != graph.ids.len()
            || self.circuit.universe_len() != graph.ids.len()
        {
            return Err(Error::new(ErrorKind::InvalidInput, "graph/state size mismatch"));
        }
        // Validate the recall threshold before any engine mutation.
        self.circuit
            .recall(&engine.active, minimum_active)
            .map_err(|_| Error::new(ErrorKind::InvalidInput, "invalid recall state"))?;
        let mut matches = Vec::with_capacity(steps);
        for _ in 0..steps {
            engine.step(graph)?;
            let recalled = self
                .observe_engine_state(engine, minimum_active)
                .map_err(|_| Error::new(ErrorKind::InvalidData, "recall counter overflow"))?;
            matches.push(recalled);
        }
        Ok(matches)
    }

    /// Aggregate streaming activation counts by caller-provided region labels.
    /// Only selected and explicitly annotated members contribute; no anatomical
    /// classification is inferred. u128 prevents overflow when summing counters.
    /// Returns (region_id, observed_activations, selected_members), sorted by ID.
    pub fn regional_activity_totals(
        &self,
        annotations: &[Option<u32>],
    ) -> Result<Vec<(u32, u128, usize)>, RecallError> {
        if annotations.len() != self.circuit.universe_len() {
            return Err(RecallError::LengthMismatch);
        }
        let mut totals = std::collections::BTreeMap::<u32, (u128, usize)>::new();
        for (&member, &count) in self.circuit.members().iter().zip(&self.counts) {
            if let Some(region) = annotations[member] {
                let entry = totals.entry(region).or_default();
                entry.0 += u128::from(count);
                entry.1 += 1;
            }
        }
        Ok(totals
            .into_iter()
            .map(|(region, (activations, members))| (region, activations, members))
            .collect())
    }

    /// Exact regional activation occupancy: (region, active_member_ticks, possible_member_ticks).
    /// The denominator includes all observed ticks for every explicitly annotated
    /// selected member, including inactive observations. No floating-point rounding
    /// or anatomical inference is performed.
    pub fn regional_occupancy(
        &self,
        annotations: &[Option<u32>],
    ) -> Result<Vec<(u32, u128, u128)>, RecallError> {
        let totals = self.regional_activity_totals(annotations)?;
        Ok(totals
            .into_iter()
            .map(|(region, active, members)| {
                (region, active, (self.ticks as u128) * (members as u128))
            })
            .collect())
    }

    /// Count inactive selected-member observations by explicitly annotated region.
    /// This is a complement of measured occupancy, not evidence of biological
    /// silence. Returns (region_id, inactive_member_ticks, possible_member_ticks).
    /// Only selected members with caller-supplied annotations contribute.
    pub fn regional_inactive_member_ticks(
        &self,
        annotations: &[Option<u32>],
    ) -> Result<Vec<(u32, u128, u128)>, RecallError> {
        self.regional_occupancy(annotations).map(|regions| {
            regions
                .into_iter()
                .map(|(region, active, possible)| (region, possible - active, possible))
                .collect()
        })
    }

    /// Select annotated regions whose observed inactive fraction meets a percentage.
    /// Requires sampled ticks; no inference is made for unobserved regions.
    /// Returns regions by descending measured inactivity fraction, then ascending ID.
    pub fn select_inactive_regions(
        &self,
        annotations: &[Option<u32>],
        minimum_inactive_percent: u8,
        minimum_members: usize,
        maximum_regions: usize,
    ) -> Result<Vec<u32>, RecallError> {
        if self.ticks == 0
            || minimum_inactive_percent > 100
            || minimum_members == 0
            || maximum_regions == 0
        {
            return Err(RecallError::ZeroBudget);
        }
        let mut candidates = self.regional_inactive_member_ticks(annotations)?;
        candidates.retain(|&(_, inactive, possible)| {
            let members = possible / u128::from(self.ticks);
            members >= minimum_members as u128
                && inactive * 100 >= possible * u128::from(minimum_inactive_percent)
        });
        // Rank by measured inactivity fraction, not by the arbitrary region ID.
        // Exact cross-products avoid floating-point rounding; region ID breaks ties.
        candidates.sort_unstable_by(|a, b| {
            (b.1 * a.2).cmp(&(a.1 * b.2)).then_with(|| a.0.cmp(&b.0))
        });
        candidates.truncate(maximum_regions);
        Ok(candidates.into_iter().map(|(region, _, _)| region).collect())
    }

    /// Construct a bounded recall circuit from the most inactive annotated region.
    /// This is an experimental recovery candidate, not an anatomical subgraph:
    /// members are selected by observed inactivity and no edges are inferred.
    /// Ties between regions and members are resolved by ascending identifiers.
    pub fn select_inactive_region_subcircuit(
        &self,
        annotations: &[Option<u32>],
        minimum_inactive_percent: u8,
        minimum_members: usize,
        maximum_members: usize,
    ) -> Result<Subcircuit, RecallError> {
        if maximum_members == 0 || minimum_members > maximum_members {
            return Err(RecallError::ZeroBudget);
        }
        let region = self
            .select_inactive_regions(annotations, minimum_inactive_percent, minimum_members, 1)?
            .into_iter()
            .next()
            .ok_or(RecallError::EmptyCircuit)?;
        let mut members: Vec<(usize, u64)> = self
            .circuit
            .members()
            .iter()
            .copied()
            .zip(self.counts.iter().copied())
            .filter(|(member, _)| annotations[*member] == Some(region))
            .collect();
        // Fewest observed activations first; deterministic index tie-break.
        members.sort_unstable_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        members.truncate(maximum_members);
        Subcircuit::new(
            self.circuit.universe_len(),
            members.into_iter().map(|(member, _)| member).collect(),
        )
    }

    /// Recall the most inactive annotated region against a live simulated state.
    /// Region and member selection use only previous observations; this read-only
    /// query does not consume a tick, stimulate the engine, or infer graph edges.
    pub fn recall_inactive_region(
        &self,
        state: &BooleanState,
        annotations: &[Option<u32>],
        minimum_inactive_percent: u8,
        minimum_members: usize,
        maximum_members: usize,
        minimum_active: usize,
    ) -> Result<(Vec<usize>, bool), RecallError> {
        if minimum_active == 0 || minimum_active > maximum_members {
            return Err(RecallError::ZeroBudget);
        }
        let selected = self.select_inactive_region_subcircuit(
            annotations,
            minimum_inactive_percent,
            minimum_members,
            maximum_members,
        )?;
        let recalled = selected.recall(state, minimum_active)?;
        Ok((selected.members().to_vec(), recalled))
    }

    /// Evaluate one historically selected inactive regional circuit across bounded
    /// future simulated states. Selection is frozen before evaluation: the future
    /// states cannot leak into region choice. Returns the first zero-based recall
    /// index, or None if no recovery occurred within the supplied observation window.
    /// This read-only experiment never advances the engine or changes counters.
    pub fn first_inactive_region_recovery(
        &self,
        states: &[BooleanState],
        annotations: &[Option<u32>],
        minimum_inactive_percent: u8,
        minimum_members: usize,
        maximum_members: usize,
        minimum_active: usize,
    ) -> Result<Option<usize>, RecallError> {
        if states.is_empty() || states.len() > 1000 || minimum_active == 0 {
            return Err(RecallError::ZeroBudget);
        }
        let selected = self.select_inactive_region_subcircuit(
            annotations,
            minimum_inactive_percent,
            minimum_members,
            maximum_members,
        )?;
        // Validate the entire window before producing a result, even when an
        // earlier state already satisfies recall.
        for state in states {
            selected.recall(state, minimum_active)?;
        }
        for (index, state) in states.iter().enumerate() {
            if selected.recall(state, minimum_active)? {
                return Ok(Some(index));
            }
        }
        Ok(None)
    }

    /// First confirmation of sustained recovery for a historically selected inactive
    /// region. The candidate is frozen before future states are inspected. A recovery
    /// requires `consecutive` matching states; the returned zero-based index is the
    /// confirmation tick, not the beginning of the streak. None means right-censored.
    /// This read-only measurement makes no anatomical or learning claim.
    pub fn first_durable_inactive_region_recovery(
        &self,
        states: &[BooleanState],
        annotations: &[Option<u32>],
        minimum_inactive_percent: u8,
        minimum_members: usize,
        maximum_members: usize,
        minimum_active: usize,
        consecutive: usize,
    ) -> Result<Option<usize>, RecallError> {
        if states.is_empty() || states.len() > 1000 || consecutive == 0
            || consecutive > states.len() || minimum_active == 0
        {
            return Err(RecallError::ZeroBudget);
        }
        let selected = self.select_inactive_region_subcircuit(
            annotations,
            minimum_inactive_percent,
            minimum_members,
            maximum_members,
        )?;
        // Validate all dimensions and thresholds, including after an early match.
        for state in states {
            selected.recall(state, minimum_active)?;
        }
        let mut streak = 0usize;
        for (index, state) in states.iter().enumerate() {
            if selected.recall(state, minimum_active)? {
                streak += 1;
                if streak == consecutive {
                    return Ok(Some(index));
                }
            } else {
                streak = 0;
            }
        }
        Ok(None)
    }

    /// Aggregate independently observed future trials for one frozen inactive-region
    /// candidate. Each trial has the same explicit observation budget; missing durable
    /// recovery is right-censored, never counted as a zero-latency recovery.
    /// Returns (confirmation_tick_histogram, censored_trials), with zero-based ticks.
    /// No future observation influences circuit selection or historical counters.
    pub fn inactive_region_recovery_histogram(
        &self,
        trials: &[Vec<BooleanState>],
        annotations: &[Option<u32>],
        minimum_inactive_percent: u8,
        minimum_members: usize,
        maximum_members: usize,
        minimum_active: usize,
        consecutive: usize,
    ) -> Result<(Vec<u64>, u64), RecallError> {
        if trials.is_empty() || trials.len() > 1000 {
            return Err(RecallError::ZeroBudget);
        }
        let budget = trials[0].len();
        if budget == 0 || budget > 1000 || consecutive == 0 || consecutive > budget {
            return Err(RecallError::ZeroBudget);
        }
        let selected = self.select_inactive_region_subcircuit(
            annotations,
            minimum_inactive_percent,
            minimum_members,
            maximum_members,
        )?;
        // Validate every trial before counting, including states after an early hit.
        for trial in trials {
            if trial.len() != budget {
                return Err(RecallError::LengthMismatch);
            }
            for state in trial {
                selected.recall(state, minimum_active)?;
            }
        }
        let mut histogram = vec![0u64; budget];
        let mut censored = 0u64;
        for trial in trials {
            let mut streak = 0usize;
            let mut confirmation = None;
            for (tick, state) in trial.iter().enumerate() {
                if selected.recall(state, minimum_active)? {
                    streak += 1;
                    if streak == consecutive {
                        confirmation = Some(tick);
                        break;
                    }
                } else {
                    streak = 0;
                }
            }
            if let Some(tick) = confirmation {
                histogram[tick] += 1;
            } else {
                censored += 1;
            }
        }
        Ok((histogram, censored))
    }

    /// Cumulative confirmed durable recoveries at each observation tick, with the
    /// number of trials still awaiting confirmation (including right-censored).
    /// Every trial uses the same horizon and the same historically frozen candidate.
    /// Returns (confirmed_by_tick, not_confirmed_by_tick); no probability estimate
    /// or biological inference is implied by these integer counts.
    pub fn inactive_region_recovery_curve(
        &self,
        trials: &[Vec<BooleanState>],
        annotations: &[Option<u32>],
        minimum_inactive_percent: u8,
        minimum_members: usize,
        maximum_members: usize,
        minimum_active: usize,
        consecutive: usize,
    ) -> Result<Vec<(u64, u64)>, RecallError> {
        let (histogram, censored) = self.inactive_region_recovery_histogram(
            trials,
            annotations,
            minimum_inactive_percent,
            minimum_members,
            maximum_members,
            minimum_active,
            consecutive,
        )?;
        let total = u64::try_from(trials.len()).map_err(|_| RecallError::ZeroBudget)?;
        let mut confirmed = 0u64;
        let mut curve = Vec::with_capacity(histogram.len());
        for hits in histogram {
            confirmed += hits; // At most 1000 trials; overflow is impossible.
            curve.push((confirmed, total - confirmed));
        }
        debug_assert_eq!(curve.last().map(|pair| pair.1), Some(censored));
        Ok(curve)
    }

    /// Per-tick discrete recovery risk table for independent, equal-horizon trials.
    /// Each row is (at_risk_at_tick, durable_confirmations_at_tick).
    /// Nonrecovering trials remain at risk through the final tick and are then
    /// right-censored. No biological hazard or causal interpretation is implied.
    pub fn inactive_region_recovery_risk_table(
        &self,
        trials: &[Vec<BooleanState>],
        annotations: &[Option<u32>],
        minimum_inactive_percent: u8,
        minimum_members: usize,
        maximum_members: usize,
        minimum_active: usize,
        consecutive: usize,
    ) -> Result<Vec<(u64, u64)>, RecallError> {
        let (histogram, censored) = self.inactive_region_recovery_histogram(
            trials,
            annotations,
            minimum_inactive_percent,
            minimum_members,
            maximum_members,
            minimum_active,
            consecutive,
        )?;
        let mut at_risk = u64::try_from(trials.len()).map_err(|_| RecallError::ZeroBudget)?;
        let mut table = Vec::with_capacity(histogram.len());
        for confirmations in histogram {
            table.push((at_risk, confirmations));
            at_risk -= confirmations;
        }
        debug_assert_eq!(at_risk, censored);
        Ok(table)
    }

    /// Compare two frozen inactive-region selection thresholds on the same future trials.
    /// Each trial has the same horizon. A censored result is not a zero-latency hit.
    /// Returns (A earlier, B earlier, equal confirmed, A only, B only, both censored).
    /// This is a paired descriptive count, not a significance test.
    pub fn compare_inactive_region_recovery(
        &self,
        trials: &[Vec<BooleanState>],
        annotations: &[Option<u32>],
        thresholds: (u8, u8),
        minimum_members: usize,
        maximum_members: usize,
        minimum_active: usize,
        consecutive: usize,
    ) -> Result<(u64, u64, u64, u64, u64, u64), RecallError> {
        if trials.is_empty() || trials.len() > 1000 {
            return Err(RecallError::ZeroBudget);
        }
        let horizon = trials[0].len();
        if horizon == 0 || horizon > 1000 || consecutive == 0 || consecutive > horizon {
            return Err(RecallError::ZeroBudget);
        }
        let a = self.select_inactive_region_subcircuit(
            annotations, thresholds.0, minimum_members, maximum_members,
        )?;
        let b = self.select_inactive_region_subcircuit(
            annotations, thresholds.1, minimum_members, maximum_members,
        )?;
        // Validate both candidates against every state, even after early recovery.
        for trial in trials {
            if trial.len() != horizon {
                return Err(RecallError::LengthMismatch);
            }
            for state in trial {
                a.recall(state, minimum_active)?;
                b.recall(state, minimum_active)?;
            }
        }
        let mut counts = (0, 0, 0, 0, 0, 0);
        for trial in trials {
            let mut streak_a = 0usize;
            let mut streak_b = 0usize;
            let mut hit_a = None;
            let mut hit_b = None;
            for (tick, state) in trial.iter().enumerate() {
                if hit_a.is_none() {
                    streak_a = if a.recall(state, minimum_active)? { streak_a + 1 } else { 0 };
                    if streak_a == consecutive { hit_a = Some(tick); }
                }
                if hit_b.is_none() {
                    streak_b = if b.recall(state, minimum_active)? { streak_b + 1 } else { 0 };
                    if streak_b == consecutive { hit_b = Some(tick); }
                }
            }
            match (hit_a, hit_b) {
                (Some(x), Some(y)) if x < y => counts.0 += 1,
                (Some(x), Some(y)) if x > y => counts.1 += 1,
                (Some(_), Some(_)) => counts.2 += 1,
                (Some(_), None) => counts.3 += 1,
                (None, Some(_)) => counts.4 += 1,
                (None, None) => counts.5 += 1,
            }
        }
        Ok(counts)
    }

    /// Per-trial paired durable recovery latency for two historically frozen regional
    /// candidates. Each pair contains zero-based confirmation ticks; None means
    /// right-censored at the common horizon. No trial changes selection or counters.
    /// This preserves pairing information lost in aggregate win/loss counts.
    pub fn paired_inactive_region_recovery_latencies(
        &self,
        trials: &[Vec<BooleanState>],
        annotations: &[Option<u32>],
        thresholds: (u8, u8),
        minimum_members: usize,
        maximum_members: usize,
        minimum_active: usize,
        consecutive: usize,
    ) -> Result<Vec<(Option<usize>, Option<usize>)>, RecallError> {
        if trials.is_empty() || trials.len() > 1000 {
            return Err(RecallError::ZeroBudget);
        }
        let horizon = trials[0].len();
        if horizon == 0 || horizon > 1000 || consecutive == 0 || consecutive > horizon {
            return Err(RecallError::ZeroBudget);
        }
        let a = self.select_inactive_region_subcircuit(
            annotations, thresholds.0, minimum_members, maximum_members,
        )?;
        let b = self.select_inactive_region_subcircuit(
            annotations, thresholds.1, minimum_members, maximum_members,
        )?;
        for trial in trials {
            if trial.len() != horizon {
                return Err(RecallError::LengthMismatch);
            }
            for state in trial {
                a.recall(state, minimum_active)?;
                b.recall(state, minimum_active)?;
            }
        }
        let mut latencies = Vec::with_capacity(trials.len());
        for trial in trials {
            let (mut streak_a, mut streak_b) = (0usize, 0usize);
            let (mut hit_a, mut hit_b) = (None, None);
            for (tick, state) in trial.iter().enumerate() {
                if hit_a.is_none() {
                    streak_a = if a.recall(state, minimum_active)? { streak_a + 1 } else { 0 };
                    if streak_a == consecutive {
                        hit_a = Some(tick);
                    }
                }
                if hit_b.is_none() {
                    streak_b = if b.recall(state, minimum_active)? { streak_b + 1 } else { 0 };
                    if streak_b == consecutive {
                        hit_b = Some(tick);
                    }
                }
                if hit_a.is_some() && hit_b.is_some() {
                    break;
                }
            }
            latencies.push((hit_a, hit_b));
        }
        Ok(latencies)
    }

    /// Choose a regional recall policy on training trials only, never held-out data.
    /// Confirmed recovery beats censoring; earlier confirmations win.
    /// Equal outcomes are non-decisive. Returns None on a tie or insufficient
    /// decisive trials; the returned index is 0 or 1 in the threshold pair.
    pub fn select_inactive_region_recovery_policy(
        &self,
        training_trials: &[Vec<BooleanState>],
        annotations: &[Option<u32>],
        thresholds: (u8, u8),
        minimum_members: usize,
        maximum_members: usize,
        minimum_active: usize,
        consecutive: usize,
        minimum_decisive: usize,
    ) -> Result<Option<u8>, RecallError> {
        if minimum_decisive == 0 || minimum_decisive > training_trials.len() {
            return Err(RecallError::ZeroBudget);
        }
        let paired = self.paired_inactive_region_recovery_latencies(
            training_trials, annotations, thresholds, minimum_members,
            maximum_members, minimum_active, consecutive,
        )?;
        let (mut wins_a, mut wins_b) = (0usize, 0usize);
        for (a, b) in paired {
            match (a, b) {
                (Some(x), Some(y)) if x < y => wins_a += 1,
                (Some(x), Some(y)) if y < x => wins_b += 1,
                (Some(_), None) => wins_a += 1,
                (None, Some(_)) => wins_b += 1,
                _ => {}
            }
        }
        if wins_a + wins_b < minimum_decisive || wins_a == wins_b {
            Ok(None)
        } else if wins_a > wins_b {
            Ok(Some(0))
        } else {
            Ok(Some(1))
        }
    }

    /// Evaluate a previously chosen policy on independent held-out trials.
    /// The choice must be made using training data only. A confirmed recovery
    /// beats a censored result; two censored results remain a separate category.
    /// Returns (wins, losses, equal confirmations, both censored).
    /// This read-only evaluation does not adjust or reselect the policy.
    pub fn evaluate_inactive_region_recovery_policy(
        &self,
        held_out_trials: &[Vec<BooleanState>],
        annotations: &[Option<u32>],
        thresholds: (u8, u8),
        minimum_members: usize,
        maximum_members: usize,
        minimum_active: usize,
        consecutive: usize,
        chosen_policy: u8,
    ) -> Result<(usize, usize, usize, usize), RecallError> {
        if chosen_policy > 1 {
            return Err(RecallError::ZeroBudget);
        }
        let paired = self.paired_inactive_region_recovery_latencies(
            held_out_trials,
            annotations,
            thresholds,
            minimum_members,
            maximum_members,
            minimum_active,
            consecutive,
        )?;
        let (mut wins, mut losses, mut equal, mut both_censored) = (0, 0, 0, 0);
        for (a, b) in paired {
            let (selected, control) = if chosen_policy == 0 { (a, b) } else { (b, a) };
            match (selected, control) {
                (Some(x), Some(y)) if x < y => wins += 1,
                (Some(x), Some(y)) if x > y => losses += 1,
                (Some(_), None) => wins += 1,
                (None, Some(_)) => losses += 1,
                (Some(_), Some(_)) => equal += 1,
                (None, None) => both_censored += 1,
            }
        }
        Ok((wins, losses, equal, both_censored))
    }

    /// Cumulative held-out policy advantage at each common observation horizon.
    /// Returns (wins, losses, ties with recovery, jointly censored) per tick.
    /// A recovery confirmed after a horizon remains censored at that horizon.
    /// The selected policy is fixed by the caller; no fitting occurs here.
    pub fn evaluate_inactive_region_policy_by_horizon(
        &self,
        held_out_trials: &[Vec<BooleanState>],
        annotations: &[Option<u32>],
        thresholds: (u8, u8),
        minimum_members: usize,
        maximum_members: usize,
        minimum_active: usize,
        consecutive: usize,
        chosen_policy: u8,
    ) -> Result<Vec<(usize, usize, usize, usize)>, RecallError> {
        if chosen_policy > 1 {
            return Err(RecallError::ZeroBudget);
        }
        let paired = self.paired_inactive_region_recovery_latencies(
            held_out_trials, annotations, thresholds, minimum_members,
            maximum_members, minimum_active, consecutive,
        )?;
        let horizon = held_out_trials.first().map_or(0, Vec::len);
        let mut result = Vec::with_capacity(horizon);
        for tick in 0..horizon {
            let (mut wins, mut losses, mut equal, mut censored) = (0, 0, 0, 0);
            for &(a, b) in &paired {
                let (selected, control) = if chosen_policy == 0 { (a, b) } else { (b, a) };
                let selected = selected.filter(|&t| t <= tick);
                let control = control.filter(|&t| t <= tick);
                match (selected, control) {
                    (Some(x), Some(y)) if x < y => wins += 1,
                    (Some(x), Some(y)) if x > y => losses += 1,
                    (Some(_), None) => wins += 1,
                    (None, Some(_)) => losses += 1,
                    (Some(_), Some(_)) => equal += 1,
                    (None, None) => censored += 1,
                }
            }
            result.push((wins, losses, equal, censored));
        }
        Ok(result)
    }

    /// Restricted total recovery latency on independent held-out trials.
    /// Confirmed recoveries cost their one-based confirmation tick; right-censored
    /// trials cost horizon + 1. The common bound avoids treating missing recovery
    /// as an immediate hit. Returns (selected_cost, control_cost) as exact integers.
    /// This is an experimental comparison, not a biological or causal estimate.
    pub fn held_out_recovery_latency_costs(
        &self,
        held_out_trials: &[Vec<BooleanState>],
        annotations: &[Option<u32>],
        thresholds: (u8, u8),
        minimum_members: usize,
        maximum_members: usize,
        minimum_active: usize,
        consecutive: usize,
        chosen_policy: u8,
    ) -> Result<(u64, u64), RecallError> {
        if chosen_policy > 1 {
            return Err(RecallError::ZeroBudget);
        }
        let paired = self.paired_inactive_region_recovery_latencies(
            held_out_trials, annotations, thresholds, minimum_members,
            maximum_members, minimum_active, consecutive,
        )?;
        let penalty = (held_out_trials[0].len() + 1) as u64;
        let mut costs = (0u64, 0u64);
        for (a, b) in paired {
            let (selected, control) = if chosen_policy == 0 { (a, b) } else { (b, a) };
            costs.0 += selected.map_or(penalty, |tick| (tick + 1) as u64);
            costs.1 += control.map_or(penalty, |tick| (tick + 1) as u64);
        }
        Ok(costs)
    }

    /// Build a recallable circuit from the most active explicitly annotated region.
    /// Candidate members must each have at least `minimum_member_ticks` activations.
    /// Region rank uses summed activity, then ascending region ID; members use
    /// descending activity, then ascending index. No graph edges are created.
    pub fn select_region_subcircuit(
        &self,
        annotations: &[Option<u32>],
        minimum_member_ticks: u64,
        maximum_members: usize,
    ) -> Result<Subcircuit, RecallError> {
        if maximum_members == 0 || minimum_member_ticks == 0 {
            return Err(RecallError::ZeroBudget);
        }
        if annotations.len() != self.circuit.universe_len() {
            return Err(RecallError::LengthMismatch);
        }
        let mut regions = std::collections::BTreeMap::<u32, u128>::new();
        for (&member, &count) in self.circuit.members().iter().zip(&self.counts) {
            if count >= minimum_member_ticks {
                if let Some(region) = annotations[member] {
                    *regions.entry(region).or_default() += u128::from(count);
                }
            }
        }
        let selected_region = regions
            .into_iter()
            .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
            .map(|(region, _)| region)
            .ok_or(RecallError::EmptyCircuit)?;
        let mut members: Vec<(usize, u64)> = self
            .circuit
            .members()
            .iter()
            .copied()
            .zip(self.counts.iter().copied())
            .filter(|&(member, count)| {
                count >= minimum_member_ticks && annotations[member] == Some(selected_region)
            })
            .collect();
        members.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        members.truncate(maximum_members);
        Subcircuit::new(
            self.circuit.universe_len(),
            members.into_iter().map(|(member, _)| member).collect(),
        )
    }

    /// Select explicitly annotated regions with sufficient observed activation.
    /// Regions are ordered by descending active-member ticks, then region ID.
    /// No biological classification or anatomical edge is inferred.
    pub fn select_active_regions(
        &self,
        annotations: &[Option<u32>],
        minimum_active_ticks: u128,
        minimum_members: usize,
        maximum_regions: usize,
    ) -> Result<Vec<u32>, RecallError> {
        if minimum_members == 0 || maximum_regions == 0 {
            return Err(RecallError::ZeroBudget);
        }
        let mut candidates = self.regional_activity_totals(annotations)?;
        candidates.retain(|(_, active, members)| {
            *active >= minimum_active_ticks && *members >= minimum_members
        });
        candidates.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        candidates.truncate(maximum_regions);
        Ok(candidates
            .into_iter()
            .map(|(region, _, _)| region)
            .collect())
    }

    /// Atomically observe one simulated state and select the most active annotated regions.
    /// Invalid annotations or budgets must not consume an observation tick.
    /// The returned match describes the current circuit recall, while region ranking
    /// uses cumulative observed activity including this tick.
    pub fn observe_and_select_regions(
        &mut self,
        state: &BooleanState,
        minimum_active: usize,
        annotations: &[Option<u32>],
        minimum_active_ticks: u128,
        minimum_members: usize,
        maximum_regions: usize,
    ) -> Result<(bool, Vec<u32>), RecallError> {
        if annotations.len() != self.circuit.universe_len() {
            return Err(RecallError::LengthMismatch);
        }
        if minimum_members == 0 || maximum_regions == 0 {
            return Err(RecallError::ZeroBudget);
        }
        let recalled = self.observe(state, minimum_active)?;
        let regions = self.select_active_regions(
            annotations,
            minimum_active_ticks,
            minimum_members,
            maximum_regions,
        )?;
        Ok((recalled, regions))
    }

    /// Recall the most active explicitly annotated region against the live engine state.
    /// Selection uses previously observed ticks; the current state is not consumed.
    /// Returns the selected member indices and whether their recall threshold is met.
    /// All graph connectivity remains unchanged and no anatomical label is inferred.
    pub fn recall_selected_region(
        &self,
        engine: &crate::engine::BooleanEngine,
        annotations: &[Option<u32>],
        minimum_member_ticks: u64,
        maximum_members: usize,
        minimum_active: usize,
    ) -> Result<(Vec<usize>, bool), RecallError> {
        if minimum_active == 0 {
            return Err(RecallError::ZeroBudget);
        }
        let selected = self.select_region_subcircuit(
            annotations,
            minimum_member_ticks,
            maximum_members,
        )?;
        let recalled = selected.recall(&engine.active, minimum_active)?;
        Ok((selected.members().to_vec(), recalled))
    }

    #[must_use]
    pub fn ticks(&self) -> u64 {
        self.ticks
    }

    #[must_use]
    pub fn latest_streak(&self) -> u64 {
        self.current_streak
    }

    #[must_use]
    pub fn longest_streak(&self) -> u64 {
        self.longest_streak
    }

    pub fn member_counts(&self) -> impl Iterator<Item = (usize, u64)> + '_ {
        self.circuit
            .members()
            .iter()
            .copied()
            .zip(self.counts.iter().copied())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn held_out_policy_horizon_curve_respects_confirmation_and_censoring() {
        let mut tracker = OnlineRecall::new(Subcircuit::new(2, vec![0, 1]).unwrap());
        let labels = [Some(1), Some(1)];
        let quiet = BooleanState::new(2);
        tracker.observe(&quiet, 1).unwrap();
        let mut active = BooleanState::new(2);
        active.set(0, true).unwrap();
        let trials = vec![vec![quiet.clone(), active.clone(), active.clone()]];
        let before = tracker.member_counts().collect::<Vec<_>>();
        let curve = tracker.evaluate_inactive_region_policy_by_horizon(
            &trials, &labels, (100, 100), 1, 2, 1, 2, 0,
        ).unwrap();
        assert_eq!(curve, vec![(0, 0, 0, 1), (0, 0, 0, 1), (0, 0, 1, 0)]);
        assert_eq!(tracker.member_counts().collect::<Vec<_>>(), before);
        assert_eq!(tracker.ticks(), 1);
        assert_eq!(
            tracker.evaluate_inactive_region_policy_by_horizon(
                &trials, &labels, (100, 100), 1, 2, 1, 2, 2,
            ),
            Err(RecallError::ZeroBudget)
        );
    }

    #[test]
    fn regional_inactive_ticks_count_only_annotated_selected_members() {
        let mut tracker = OnlineRecall::new(Subcircuit::new(4, vec![0, 1, 3]).unwrap());
        let labels = [Some(7), Some(7), Some(9), None];
        assert_eq!(tracker.regional_inactive_member_ticks(&labels), Ok(vec![(7, 0, 0)]));
        let mut first = BooleanState::new(4);
        first.set(0, true).unwrap();
        first.set(2, true).unwrap(); // Not selected; must not contribute.
        tracker.observe(&first, 1).unwrap();
        let second = BooleanState::new(4);
        tracker.observe(&second, 1).unwrap();
        assert_eq!(tracker.regional_inactive_member_ticks(&labels), Ok(vec![(7, 3, 4)]));
        assert_eq!(tracker.ticks(), 2);
        assert_eq!(tracker.regional_inactive_member_ticks(&labels[..3]), Err(RecallError::LengthMismatch));
    }


    #[test]
    fn inactive_region_selection_ranks_by_fraction_before_region_id() {
        let mut tracker = OnlineRecall::new(Subcircuit::new(4, vec![0, 1, 2, 3]).unwrap());
        let labels = [Some(2), Some(2), Some(9), Some(9)];
        let mut state = BooleanState::new(4);
        state.set(0, true).unwrap();
        tracker.observe(&state, 1).unwrap();
        // Region 2: 1/2 inactive; region 9: 2/2 inactive.
        assert_eq!(tracker.select_inactive_regions(&labels, 50, 2, 1), Ok(vec![9]));
        assert_eq!(tracker.select_inactive_regions(&labels, 50, 2, 2), Ok(vec![9, 2]));
        assert_eq!(tracker.ticks(), 1);
    }

    #[test]
    fn inactive_region_recovery_risk_table_preserves_end_censoring() {
        let mut tracker = OnlineRecall::new(Subcircuit::new(2, vec![0, 1]).unwrap());
        let labels = [Some(5), Some(5)];
        let quiet = BooleanState::new(2);
        tracker.observe(&quiet, 1).unwrap();
        let mut active = BooleanState::new(2);
        active.set(0, true).unwrap();
        let trials = vec![
            vec![active.clone(), active.clone(), quiet.clone()],
            vec![quiet.clone(), active.clone(), active.clone()],
            vec![quiet.clone(), quiet.clone(), quiet.clone()],
        ];
        assert_eq!(
            tracker.inactive_region_recovery_risk_table(&trials, &labels, 100, 1, 2, 1, 2),
            Ok(vec![(3, 0), (3, 1), (2, 1)])
        );
        assert_eq!(
            tracker.inactive_region_recovery_risk_table(&trials, &labels, 100, 1, 2, 1, 0),
            Err(RecallError::ZeroBudget)
        );
        assert_eq!(tracker.ticks(), 1);
        assert_eq!(tracker.member_counts().collect::<Vec<_>>(), vec![(0, 0), (1, 0)]);
    }

    #[test]
    fn inactive_region_recovery_curve_preserves_censored_trials() {
        let mut tracker = OnlineRecall::new(Subcircuit::new(2, vec![0, 1]).unwrap());
        let annotations = [Some(7), Some(7)];
        let quiet = BooleanState::new(2);
        tracker.observe(&quiet, 1).unwrap();
        let mut active = BooleanState::new(2);
        active.set(0, true).unwrap();
        let trials = vec![
            vec![quiet.clone(), active.clone(), active.clone()],
            vec![active.clone(), quiet.clone(), active.clone()],
            vec![quiet.clone(), quiet.clone(), quiet.clone()],
        ];
        let before = tracker.member_counts().collect::<Vec<_>>();
        assert_eq!(
            tracker.inactive_region_recovery_curve(&trials, &annotations, 100, 1, 2, 1, 2),
            Ok(vec![(0, 3), (0, 3), (1, 2)])
        );
        assert_eq!(tracker.member_counts().collect::<Vec<_>>(), before);
        assert_eq!(tracker.ticks(), 1);
        assert_eq!(
            tracker.inactive_region_recovery_curve(&trials, &annotations[..1], 100, 1, 2, 1, 2),
            Err(RecallError::LengthMismatch)
        );
    }

    #[test]
    fn inactive_region_histogram_counts_confirmations_and_censoring() {
        let mut tracker = OnlineRecall::new(Subcircuit::new(2, vec![0, 1]).unwrap());
        let annotations = [Some(7), Some(7)];
        let quiet = BooleanState::new(2);
        tracker.observe(&quiet, 1).unwrap();
        let mut active = BooleanState::new(2);
        active.set(0, true).unwrap();
        let trials = vec![
            vec![quiet.clone(), active.clone(), active.clone()],
            vec![active.clone(), quiet.clone(), active.clone()],
            vec![quiet.clone(), quiet.clone(), quiet.clone()],
        ];
        let before = tracker.ticks();
        assert_eq!(
            tracker.inactive_region_recovery_histogram(
                &trials, &annotations, 100, 1, 2, 1, 2
            ),
            Ok((vec![0, 0, 1], 2))
        );
        assert_eq!(tracker.ticks(), before);
        let mut invalid = trials.clone();
        invalid[2].pop();
        assert_eq!(
            tracker.inactive_region_recovery_histogram(
                &invalid, &annotations, 100, 1, 2, 1, 2
            ),
            Err(RecallError::LengthMismatch)
        );
    }

    #[test]
    fn recalls_selected_region_against_live_engine_without_consuming_ticks() {
        use crate::connectome::Connectome;
        use crate::engine::BooleanEngine;

        // Synthetic chain; region IDs are explicit experimental labels.
        let graph = Connectome {
            ids: vec![10, 20, 30],
            offsets: vec![0, 1, 2, 2],
            targets: vec![1, 2],
            counts: vec![1, 1],
        };
        let mut engine = BooleanEngine::new(&graph, 1).unwrap();
        let mut tracker = OnlineRecall::new(Subcircuit::new(3, vec![0, 1, 2]).unwrap());
        let regions = [Some(7), Some(7), Some(9)];
        engine.stimulate(&graph, 10).unwrap();
        tracker.observe_engine_state(&engine, 1).unwrap();
        engine.step(&graph).unwrap();
        tracker.observe_engine_state(&engine, 1).unwrap();
        assert_eq!(
            tracker.recall_selected_region(&engine, &regions, 1, 2, 1),
            Ok((vec![0, 1], true))
        );
        engine.step(&graph).unwrap();
        assert_eq!(
            tracker.recall_selected_region(&engine, &regions, 1, 2, 1),
            Ok((vec![0, 1], false))
        );
        assert_eq!(tracker.ticks(), 2);
        assert_eq!(
            tracker.recall_selected_region(&engine, &regions[..2], 1, 2, 1),
            Err(RecallError::LengthMismatch)
        );
        assert_eq!(
            tracker.recall_selected_region(&engine, &regions, 1, 2, 0),
            Err(RecallError::ZeroBudget)
        );
    }



    #[test]
    fn streams_without_retaining_history_and_preserves_member_order() {
        let circuit = Subcircuit::new(5, vec![4, 1, 3]).unwrap();
        let mut tracker = OnlineRecall::new(circuit);
        let mut active = BooleanState::new(5);
        active.set(1, true).unwrap();
        active.set(4, true).unwrap();
        let quiet = BooleanState::new(5);
        assert_eq!(tracker.observe(&active, 2), Ok(true));
        assert_eq!(tracker.observe(&active, 2), Ok(true));
        assert_eq!(tracker.observe(&quiet, 2), Ok(false));
        assert_eq!(tracker.observe(&active, 2), Ok(true));
        assert_eq!(tracker.ticks(), 4);
        assert_eq!(tracker.latest_streak(), 1);
        assert_eq!(tracker.longest_streak(), 2);
        assert_eq!(
            tracker.member_counts().collect::<Vec<_>>(),
            vec![(1, 3), (3, 0), (4, 3)]
        );
    }

    #[test]
    fn region_totals_stream_without_anatomical_inference() {
        let circuit = Subcircuit::new(5, vec![0, 1, 3, 4]).unwrap();
        let mut tracker = OnlineRecall::new(circuit);
        let labels = [Some(9), Some(2), Some(2), Some(9), None];
        assert_eq!(
            tracker.regional_activity_totals(&labels),
            Ok(vec![(2, 0, 1), (9, 0, 2)])
        );
        let mut first = BooleanState::new(5);
        first.set(0, true).unwrap();
        first.set(1, true).unwrap();
        first.set(2, true).unwrap();
        tracker.observe(&first, 1).unwrap();
        let mut second = BooleanState::new(5);
        second.set(3, true).unwrap();
        second.set(4, true).unwrap();
        tracker.observe(&second, 1).unwrap();
        assert_eq!(
            tracker.regional_activity_totals(&labels),
            Ok(vec![(2, 1, 1), (9, 2, 2)])
        );
        assert_eq!(
            tracker.regional_activity_totals(&labels[..4]),
            Err(RecallError::LengthMismatch)
        );
        assert_eq!(tracker.ticks(), 2);
    }

    #[test]
    fn observes_actual_engine_steps_without_retaining_snapshots() {
        use crate::connectome::Connectome;
        use crate::engine::BooleanEngine;

        // Synthetic directed chain 10 -> 20 -> 30. No biological claims.
        let graph = Connectome {
            ids: vec![10, 20, 30],
            offsets: vec![0, 1, 2, 2],
            targets: vec![1, 2],
            counts: vec![1, 1],
        };
        let mut engine = BooleanEngine::new(&graph, 1).unwrap();
        let mut tracker = OnlineRecall::new(Subcircuit::new(3, vec![0, 1, 2]).unwrap());
        engine.stimulate(&graph, 10).unwrap();
        assert_eq!(tracker.observe_engine_state(&engine, 1), Ok(true));
        assert_eq!(
            tracker.member_counts().collect::<Vec<_>>(),
            vec![(0, 1), (1, 0), (2, 0)]
        );
        engine.step(&graph).unwrap();
        assert_eq!(tracker.observe_engine_state(&engine, 1), Ok(true));
        engine.step(&graph).unwrap();
        assert_eq!(tracker.observe_engine_state(&engine, 1), Ok(true));
        engine.step(&graph).unwrap();
        assert_eq!(tracker.observe_engine_state(&engine, 1), Ok(false));
        assert_eq!(tracker.ticks(), 4);
        assert_eq!(tracker.latest_streak(), 0);
        assert_eq!(tracker.longest_streak(), 3);
        assert_eq!(
            tracker.member_counts().collect::<Vec<_>>(),
            vec![(0, 1), (1, 1), (2, 1)]
        );
        assert_eq!(
            tracker.regional_activity_totals(&[Some(4), Some(4), Some(8)]),
            Ok(vec![(4, 2, 2), (8, 1, 1)])
        );
    }

    #[test]
    fn regional_occupancy_counts_inactive_ticks_and_unannotated_members() {
        let mut tracker = OnlineRecall::new(Subcircuit::new(4, vec![0, 1, 3]).unwrap());
        let labels = [Some(7), Some(7), Some(9), None];
        assert_eq!(tracker.regional_occupancy(&labels), Ok(vec![(7, 0, 0)]));
        let mut first = BooleanState::new(4);
        first.set(0, true).unwrap();
        first.set(3, true).unwrap();
        tracker.observe(&first, 1).unwrap();
        tracker.observe(&BooleanState::new(4), 1).unwrap();
        assert_eq!(tracker.regional_occupancy(&labels), Ok(vec![(7, 1, 4)]));
        assert_eq!(
            tracker.regional_occupancy(&labels[..3]),
            Err(RecallError::LengthMismatch)
        );
        assert_eq!(tracker.ticks(), 2);
    }

    #[test]
    fn selects_recallable_region_from_streamed_activity() {
        let mut tracker = OnlineRecall::new(Subcircuit::new(5, vec![0, 1, 2, 4]).unwrap());
        let labels = [Some(7), Some(7), Some(9), None, Some(9)];
        let mut first = BooleanState::new(5);
        first.set(0, true).unwrap();
        first.set(1, true).unwrap();
        first.set(2, true).unwrap();
        tracker.observe(&first, 1).unwrap();
        let mut second = BooleanState::new(5);
        second.set(0, true).unwrap();
        second.set(4, true).unwrap();
        tracker.observe(&second, 1).unwrap();
        let selected = tracker.select_region_subcircuit(&labels, 1, 2).unwrap();
        assert_eq!(selected.members(), &[0, 1]);
        assert_eq!(selected.recall(&first, 2), Ok(true));
        assert_eq!(selected.recall(&second, 2), Ok(false));
        assert_eq!(
            tracker.select_region_subcircuit(&labels, 3, 2),
            Err(RecallError::EmptyCircuit)
        );
        assert_eq!(
            tracker.select_region_subcircuit(&labels[..4], 1, 2),
            Err(RecallError::LengthMismatch)
        );
        assert_eq!(
            tracker.select_region_subcircuit(&labels, 1, 0),
            Err(RecallError::ZeroBudget)
        );
    }

    #[test]
    fn selects_annotated_regions_by_streamed_activity_with_bounded_budget() {
        let mut tracker = OnlineRecall::new(Subcircuit::new(4, vec![0, 1, 2, 3]).unwrap());
        let labels = [Some(9), Some(9), Some(2), None];
        let mut state = BooleanState::new(4);
        state.set(0, true).unwrap();
        state.set(2, true).unwrap();
        state.set(3, true).unwrap();
        tracker.observe(&state, 1).unwrap();
        assert_eq!(
            tracker.select_active_regions(&labels, 1, 1, 2),
            Ok(vec![2, 9])
        );
        assert_eq!(tracker.select_active_regions(&labels, 1, 2, 2), Ok(vec![9]));
        assert_eq!(tracker.select_active_regions(&labels, 2, 1, 2), Ok(vec![]));
        assert_eq!(tracker.select_active_regions(&labels, 0, 1, 1), Ok(vec![2]));
        assert_eq!(
            tracker.select_active_regions(&labels, 0, 0, 1),
            Err(RecallError::ZeroBudget)
        );
        assert_eq!(
            tracker.select_active_regions(&labels[..3], 0, 1, 1),
            Err(RecallError::LengthMismatch)
        );
    }

    #[test]
    fn observes_and_selects_regions_atomically() {
        let mut tracker = OnlineRecall::new(Subcircuit::new(4, vec![0, 1, 2]).unwrap());
        let labels = [Some(8), Some(8), Some(3), None];
        let mut state = BooleanState::new(4);
        state.set(0, true).unwrap();
        state.set(2, true).unwrap();
        assert_eq!(
            tracker.observe_and_select_regions(&state, 2, &labels, 1, 1, 2),
            Ok((true, vec![3, 8]))
        );
        assert_eq!(tracker.ticks(), 1);
        assert_eq!(
            tracker.observe_and_select_regions(&state, 2, &labels[..3], 1, 1, 2),
            Err(RecallError::LengthMismatch)
        );
        assert_eq!(
            tracker.observe_and_select_regions(&state, 2, &labels, 1, 0, 2),
            Err(RecallError::ZeroBudget)
        );
        assert_eq!(
            tracker.observe_and_select_regions(&BooleanState::new(3), 2, &labels, 1, 1, 2),
            Err(RecallError::LengthMismatch)
        );
        assert_eq!(tracker.ticks(), 1);
        assert_eq!(
            tracker.member_counts().collect::<Vec<_>>(),
            vec![(0, 1), (1, 0), (2, 1)]
        );
        assert_eq!(
            tracker.observe_and_select_regions(&BooleanState::new(4), 2, &labels, 2, 1, 2),
            Ok((false, vec![]))
        );
        assert_eq!(tracker.ticks(), 2);
    }

    #[test]
    fn bounded_engine_steps_observe_propagation_and_reject_bad_budgets() {
        use crate::connectome::Connectome;
        use crate::engine::BooleanEngine;
        let graph = Connectome {
            ids: vec![10, 20, 30],
            offsets: vec![0, 1, 2, 2],
            targets: vec![1, 2],
            counts: vec![1, 1],
        };
        let mut engine = BooleanEngine::new(&graph, 1).unwrap();
        let mut tracker = OnlineRecall::new(Subcircuit::new(3, vec![1, 2]).unwrap());
        engine.stimulate(&graph, 10).unwrap();
        assert!(tracker.observe_engine_steps(&mut engine, &graph, 1, 0).is_err());
        assert!(tracker.observe_engine_steps(&mut engine, &graph, 1, 1001).is_err());
        assert!(tracker.observe_engine_steps(&mut engine, &graph, 0, 1).is_err());
        assert_eq!(tracker.ticks(), 0);
        assert_eq!(
            tracker.observe_engine_steps(&mut engine, &graph, 1, 3).unwrap(),
            vec![true, true, false]
        );
        assert_eq!(tracker.ticks(), 3);
        assert_eq!(tracker.longest_streak(), 2);
        assert_eq!(tracker.latest_streak(), 0);
        assert_eq!(
            tracker.member_counts().collect::<Vec<_>>(),
            vec![(1, 1), (2, 1)]
        );
        assert_eq!(
            tracker.regional_occupancy(&[None, Some(7), Some(7)]),
            Ok(vec![(7, 2, 6)])
        );
    }

    #[test]
    fn inactive_region_selection_uses_only_observed_annotated_members() {
        let mut tracker = OnlineRecall::new(Subcircuit::new(4, vec![0, 1, 2, 3]).unwrap());
        let annotations = [Some(7), Some(7), Some(3), None];
        assert_eq!(
            tracker.select_inactive_regions(&annotations, 50, 1, 2),
            Err(RecallError::ZeroBudget)
        );
        let mut state = BooleanState::new(4);
        state.set(0, true).unwrap();
        tracker.observe(&state, 1).unwrap();
        assert_eq!(tracker.select_inactive_regions(&annotations, 50, 1, 2), Ok(vec![3, 7]));
        assert_eq!(tracker.select_inactive_regions(&annotations, 100, 1, 2), Ok(vec![3]));
        assert_eq!(tracker.select_inactive_regions(&annotations, 50, 2, 2), Ok(vec![7]));
        assert_eq!(tracker.select_inactive_regions(&annotations, 50, 1, 1), Ok(vec![3]));
        assert_eq!(
            tracker.select_inactive_regions(&annotations[..3], 50, 1, 2),
            Err(RecallError::LengthMismatch)
        );
        assert_eq!(tracker.ticks(), 1);
    }

    #[test]
    fn inactive_region_subcircuit_prefers_quiet_members_without_edges() {
        let mut tracker = OnlineRecall::new(Subcircuit::new(5, vec![0, 1, 2, 3]).unwrap());
        let labels = [Some(8), Some(8), Some(8), Some(3), None];
        let mut state = BooleanState::new(5);
        state.set(0, true).unwrap();
        state.set(3, true).unwrap();
        tracker.observe(&state, 1).unwrap();
        let selected = tracker.select_inactive_region_subcircuit(&labels, 50, 2, 2).unwrap();
        assert_eq!(selected.members(), &[1, 2]);
        // Never return a circuit smaller than the caller's minimum budget.
        assert_eq!(
            tracker.select_inactive_region_subcircuit(&labels, 50, 3, 2),
            Err(RecallError::ZeroBudget)
        );
        assert_eq!(tracker.ticks(), 1);
        assert_eq!(
            tracker.select_inactive_region_subcircuit(&labels, 100, 2, 2),
            Err(RecallError::EmptyCircuit)
        );
        assert_eq!(
            tracker.select_inactive_region_subcircuit(&labels[..4], 50, 2, 2),
            Err(RecallError::LengthMismatch)
        );
    }

    #[test]
    fn inactive_region_recall_is_read_only_and_bounded() {
        let mut tracker = OnlineRecall::new(Subcircuit::new(4, vec![0, 1, 2]).unwrap());
        let labels = [Some(7), Some(7), Some(3), None];
        let mut observed = BooleanState::new(4);
        observed.set(2, true).unwrap();
        tracker.observe(&observed, 1).unwrap();
        let mut live = BooleanState::new(4);
        live.set(0, true).unwrap();
        live.set(1, true).unwrap();
        assert_eq!(
            tracker.recall_inactive_region(&live, &labels, 100, 2, 2, 2),
            Ok((vec![0, 1], true))
        );
        assert_eq!(
            tracker.recall_inactive_region(&observed, &labels, 100, 2, 2, 1),
            Ok((vec![0, 1], false))
        );
        assert_eq!(
            tracker.recall_inactive_region(&live, &labels, 100, 2, 2, 3),
            Err(RecallError::ZeroBudget)
        );
        assert_eq!(
            tracker.recall_inactive_region(&live, &labels[..3], 100, 2, 2, 1),
            Err(RecallError::LengthMismatch)
        );
        assert_eq!(tracker.ticks(), 1);
    }

    #[test]
    fn durable_inactive_recovery_requires_consecutive_matches() {
        let mut tracker = OnlineRecall::new(Subcircuit::new(3, vec![0, 1, 2]).unwrap());
        let labels = [Some(4), Some(4), Some(9)];
        tracker.observe(&BooleanState::new(3), 1).unwrap();
        let mut active = BooleanState::new(3);
        active.set(0, true).unwrap();
        active.set(1, true).unwrap();
        let quiet = BooleanState::new(3);
        let states = [active.clone(), quiet, active.clone(), active.clone()];
        assert_eq!(tracker.first_durable_inactive_region_recovery(
            &states, &labels, 100, 2, 2, 2, 2), Ok(Some(3)));
        assert_eq!(tracker.first_durable_inactive_region_recovery(
            &states[..3], &labels, 100, 2, 2, 2, 2), Ok(None));
        assert_eq!(tracker.first_durable_inactive_region_recovery(
            &states, &labels, 100, 2, 2, 2, 0), Err(RecallError::ZeroBudget));
        assert_eq!(tracker.first_durable_inactive_region_recovery(
            &states, &labels[..2], 100, 2, 2, 2, 2), Err(RecallError::LengthMismatch));
        assert_eq!(tracker.ticks(), 1);
    }

    #[test]
    fn paired_recovery_comparison_preserves_censoring_and_budgets() {
        let mut tracker = OnlineRecall::new(Subcircuit::new(4, vec![0, 1, 2, 3]).unwrap());
        let labels = [Some(1), Some(1), Some(2), Some(2)];
        let mut historical = BooleanState::new(4);
        historical.set(2, true).unwrap();
        tracker.observe(&historical, 1).unwrap();
        let mut recovered = BooleanState::new(4);
        recovered.set(0, true).unwrap();
        recovered.set(1, true).unwrap();
        let trials = vec![
            vec![recovered.clone(), recovered.clone()],
            vec![BooleanState::new(4), BooleanState::new(4)],
        ];
        // Both thresholds choose the same fully inactive region.
        assert_eq!(
            tracker.compare_inactive_region_recovery(&trials, &labels, (50, 100), 2, 2, 2, 2),
            Ok((0, 0, 1, 0, 0, 1))
        );
        assert_eq!(
            tracker.compare_inactive_region_recovery(&trials[..1], &labels, (50, 100), 2, 2, 2, 3),
            Err(RecallError::ZeroBudget)
        );
        assert_eq!(
            tracker.compare_inactive_region_recovery(&trials, &labels[..3], (50, 100), 2, 2, 2, 2),
            Err(RecallError::LengthMismatch)
        );
        assert_eq!(tracker.ticks(), 1);
    }

    #[test]
    fn paired_latencies_preserve_trial_order_and_right_censoring() {
        let mut tracker = OnlineRecall::new(Subcircuit::new(4, vec![0, 1, 2, 3]).unwrap());
        let labels = [Some(1), Some(1), Some(2), Some(2)];
        let mut historical = BooleanState::new(4);
        historical.set(2, true).unwrap();
        tracker.observe(&historical, 1).unwrap();
        let mut active = BooleanState::new(4);
        active.set(0, true).unwrap();
        active.set(1, true).unwrap();
        let quiet = BooleanState::new(4);
        let trials = vec![
            vec![active.clone(), active.clone(), quiet.clone()],
            vec![quiet.clone(), quiet.clone(), quiet.clone()],
            vec![quiet.clone(), active.clone(), active],
        ];
        assert_eq!(
            tracker.paired_inactive_region_recovery_latencies(
                &trials, &labels, (50, 100), 2, 2, 2, 2,
            ),
            Ok(vec![(Some(1), Some(1)), (None, None), (Some(2), Some(2))])
        );
        assert_eq!(
            tracker.paired_inactive_region_recovery_latencies(
                &trials[..2], &labels, (50, 100), 2, 2, 2, 4,
            ),
            Err(RecallError::ZeroBudget)
        );
        assert_eq!(
            tracker.paired_inactive_region_recovery_latencies(
                &trials, &labels[..3], (50, 100), 2, 2, 2, 2,
            ),
            Err(RecallError::LengthMismatch)
        );
        assert_eq!(tracker.ticks(), 1);
    }

    #[test]
    fn training_policy_rejects_ties_and_insufficient_evidence() {
        let mut tracker = OnlineRecall::new(Subcircuit::new(4, vec![0, 1, 2, 3]).unwrap());
        let labels = [Some(1), Some(1), Some(2), Some(2)];
        let mut historical = BooleanState::new(4);
        historical.set(2, true).unwrap();
        tracker.observe(&historical, 1).unwrap();
        let mut recovered = BooleanState::new(4);
        recovered.set(0, true).unwrap();
        recovered.set(1, true).unwrap();
        let quiet = BooleanState::new(4);
        let training = vec![
            vec![recovered.clone(), recovered],
            vec![quiet.clone(), quiet],
        ];
        assert_eq!(
            tracker.select_inactive_region_recovery_policy(
                &training, &labels, (50, 100), 2, 2, 2, 2, 1,
            ),
            Ok(None)
        );
        assert_eq!(
            tracker.select_inactive_region_recovery_policy(
                &training, &labels, (50, 100), 2, 2, 2, 2, 0,
            ),
            Err(RecallError::ZeroBudget)
        );
        assert_eq!(
            tracker.select_inactive_region_recovery_policy(
                &training, &labels, (50, 100), 2, 2, 2, 2, 3,
            ),
            Err(RecallError::ZeroBudget)
        );
        assert_eq!(tracker.ticks(), 1);
    }

    #[test]
    fn held_out_policy_evaluation_keeps_censoring_and_history() {
        let mut tracker = OnlineRecall::new(Subcircuit::new(4, vec![0, 1, 2, 3]).unwrap());
        let labels = [Some(1), Some(1), Some(2), Some(2)];
        let mut historical = BooleanState::new(4);
        historical.set(2, true).unwrap();
        tracker.observe(&historical, 1).unwrap();
        let mut recovered = BooleanState::new(4);
        recovered.set(0, true).unwrap();
        recovered.set(1, true).unwrap();
        let quiet = BooleanState::new(4);
        let held_out = vec![
            vec![recovered.clone(), recovered],
            vec![quiet.clone(), quiet],
        ];
        assert_eq!(
            tracker.evaluate_inactive_region_recovery_policy(
                &held_out, &labels, (50, 100), 2, 2, 2, 2, 0,
            ),
            Ok((0, 0, 1, 1))
        );
        assert_eq!(
            tracker.evaluate_inactive_region_recovery_policy(
                &held_out, &labels, (50, 100), 2, 2, 2, 2, 1,
            ),
            Ok((0, 0, 1, 1))
        );
        assert_eq!(
            tracker.evaluate_inactive_region_recovery_policy(
                &held_out, &labels, (50, 100), 2, 2, 2, 2, 2,
            ),
            Err(RecallError::ZeroBudget)
        );
        assert_eq!(
            tracker.evaluate_inactive_region_recovery_policy(
                &held_out, &labels[..3], (50, 100), 2, 2, 2, 2, 0,
            ),
            Err(RecallError::LengthMismatch)
        );
        assert_eq!(tracker.ticks(), 1);
    }

    #[test]
    fn held_out_latency_costs_penalize_censoring_without_changing_history() {
        let mut tracker = OnlineRecall::new(Subcircuit::new(4, vec![0, 1, 2, 3]).unwrap());
        let labels = [Some(1), Some(1), Some(2), Some(2)];
        let mut historical = BooleanState::new(4);
        historical.set(2, true).unwrap();
        tracker.observe(&historical, 1).unwrap();
        let mut active = BooleanState::new(4);
        active.set(0, true).unwrap();
        active.set(1, true).unwrap();
        let quiet = BooleanState::new(4);
        let trials = vec![
            vec![active.clone(), active, quiet.clone()],
            vec![quiet.clone(), quiet.clone(), quiet],
        ];
        // Both thresholds select the same region: first hit at tick 1
        // costs 2, while a censored three-tick trial costs 4.
        assert_eq!(
            tracker.held_out_recovery_latency_costs(
                &trials, &labels, (50, 100), 2, 2, 2, 2, 0,
            ),
            Ok((6, 6))
        );
        assert_eq!(
            tracker.held_out_recovery_latency_costs(
                &trials, &labels, (50, 100), 2, 2, 2, 2, 1,
            ),
            Ok((6, 6))
        );
        assert_eq!(
            tracker.held_out_recovery_latency_costs(
                &trials, &labels, (50, 100), 2, 2, 2, 2, 2,
            ),
            Err(RecallError::ZeroBudget)
        );
        assert_eq!(tracker.ticks(), 1);
    }

    #[test]
    fn rejects_bad_input_without_mutation() {
        let mut tracker = OnlineRecall::new(Subcircuit::new(3, vec![0]).unwrap());
        assert_eq!(
            tracker.observe(&BooleanState::new(2), 1),
            Err(RecallError::LengthMismatch)
        );
        assert_eq!(
            tracker.observe(&BooleanState::new(3), 0),
            Err(RecallError::ZeroBudget)
        );
        assert_eq!(tracker.ticks(), 0);
        assert_eq!(tracker.member_counts().collect::<Vec<_>>(), vec![(0, 0)]);
    }
}
