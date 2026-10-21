//! Exact diagonal state in a tagged ring, inspired by NCBI's diagonal arrays.
//! For ascending subject positions, diagonals sharing a slot cannot both recur
//! when ring length >= the number of possible query seed positions. Keep the tag
//! anyway: replacing an expired diagonal must reset both one-hit and two-hit state.
use std::collections::HashMap;

const RING_BUDGET_BYTES: usize = 2 * 1024 * 1024;
const MAX_RING_ENTRIES: usize = 1 << 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct DiagonalState {
    last: usize,
    pub covered_until: usize,
}
impl Default for DiagonalState {
    fn default() -> Self {
        Self {
            last: usize::MAX,
            covered_until: 0,
        }
    }
}
impl DiagonalState {
    pub fn last_hit(&self) -> Option<usize> {
        (self.last != usize::MAX).then_some(self.last)
    }
    pub fn set_last_hit(&mut self, position: usize) {
        self.last = position;
    }
}

#[derive(Clone, Copy, Default)]
pub(super) struct Slot {
    tag: i64,
    generation: u32,
    state: DiagonalState,
}

pub(super) enum DiagonalTracker {
    Ring { slots: Vec<Slot>, generation: u32 },
    Sparse(HashMap<i64, DiagonalState>),
}
impl Default for DiagonalTracker {
    fn default() -> Self {
        Self::Sparse(HashMap::new())
    }
}
impl DiagonalTracker {
    fn configure(&mut self, seed_positions: usize, remaining_bytes: &mut usize) {
        let size = seed_positions.max(1).checked_next_power_of_two();
        let size =
            size.filter(|&n| n <= MAX_RING_ENTRIES && n <= *remaining_bytes / size_of::<Slot>());
        if let Some(size) = size {
            *remaining_bytes -= size * size_of::<Slot>();
            if !matches!(self, Self::Ring { slots, .. } if slots.len() == size) {
                *self = Self::Ring {
                    slots: vec![Slot::default(); size],
                    generation: 0,
                };
            }
        } else if !matches!(self, Self::Sparse(_)) {
            *self = Self::Sparse(HashMap::new());
        }
    }

    pub fn next_subject(&mut self) {
        match self {
            Self::Ring { slots, generation } => {
                *generation = generation.wrapping_add(1);
                if *generation == 0 {
                    slots.fill(Slot::default());
                    *generation = 1;
                }
            }
            Self::Sparse(states) => states.clear(),
        }
    }

    /// Caller visits one context in ascending subject order, with query seed
    /// positions in 0..seed_positions. No state is shared across contexts/subjects.
    #[inline]
    pub fn entry(&mut self, subject: usize, query: usize) -> &mut DiagonalState {
        let diagonal = subject as i64 - query as i64;
        match self {
            Self::Sparse(states) => states.entry(diagonal).or_default(),
            Self::Ring { slots, generation } => {
                let index = subject.wrapping_sub(query) & (slots.len() - 1);
                let slot = &mut slots[index];
                if slot.generation != *generation || slot.tag != diagonal {
                    *slot = Slot {
                        tag: diagonal,
                        generation: *generation,
                        state: DiagonalState::default(),
                    };
                }
                &mut slot.state
            }
        }
    }
}

/// Worker-owned reusable seed state. Dense rings use at most 2 MiB in total.
/// Large query contexts fall back to exact sparse state, not lossy modulo aliases.
#[derive(Default)]
pub struct SeedScratch {
    pub(super) diagonals: Vec<DiagonalTracker>,
}
impl SeedScratch {
    pub(super) fn configure(&mut self, positions: impl ExactSizeIterator<Item = usize>) {
        self.configure_with_budget(positions, RING_BUDGET_BYTES);
    }

    fn configure_with_budget(
        &mut self,
        positions: impl ExactSizeIterator<Item = usize>,
        mut budget: usize,
    ) {
        self.diagonals
            .resize_with(positions.len(), DiagonalTracker::default);
        for (tracker, positions) in self.diagonals.iter_mut().zip(positions) {
            tracker.configure(positions, &mut budget);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_and_sparse_match_unbounded_maps_across_collisions_and_subjects() {
        for positions in [1, 2, 3, 31, 32, 33, 127, 257] {
            for budget in [0, RING_BUDGET_BYTES] {
                let mut scratch = SeedScratch::default();
                scratch.configure_with_budget([positions].into_iter(), budget);
                let tracker = &mut scratch.diagonals[0];
                for subject in 0..4 {
                    tracker.next_subject();
                    let mut expected = HashMap::<i64, DiagonalState>::new();
                    for s in 0..positions * 4 + 3 {
                        // Reverse posting order also preserves ring correctness.
                        for q in (0..positions).rev().filter(|q| (s + q + subject) % 3 != 0) {
                            let reference = expected.entry(s as i64 - q as i64).or_default();
                            let actual = tracker.entry(s, q);
                            assert_eq!(actual, reference, "positions={positions}, s={s}, q={q}");
                            if (s + q) % 2 == 0 {
                                actual.set_last_hit(s);
                                reference.set_last_hit(s);
                            }
                            actual.covered_until = s + q + 17;
                            reference.covered_until = actual.covered_until;
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn epoch_wrap_budget_fallback_and_reconfiguration() {
        let mut scratch = SeedScratch::default();
        let one_ring = 4 * size_of::<Slot>();
        scratch.configure_with_budget([3, 3, MAX_RING_ENTRIES + 1].into_iter(), one_ring);
        assert!(matches!(scratch.diagonals[0], DiagonalTracker::Ring { .. }));
        assert!(matches!(scratch.diagonals[1], DiagonalTracker::Sparse(_)));
        assert!(matches!(scratch.diagonals[2], DiagonalTracker::Sparse(_)));
        let tracker = &mut scratch.diagonals[0];
        tracker.next_subject();
        tracker.entry(0, 0).set_last_hit(0);
        assert_eq!(tracker.entry(0, 0).last_hit(), Some(0));
        if let DiagonalTracker::Ring { generation, .. } = tracker {
            *generation = u32::MAX;
        }
        tracker.next_subject();
        assert_eq!(*tracker.entry(0, 0), DiagonalState::default());
        scratch.configure([1000].into_iter());
        assert_eq!(scratch.diagonals.len(), 1);
        scratch.diagonals[0].next_subject();
        assert_eq!(*scratch.diagonals[0].entry(2, 31), DiagonalState::default());
        scratch.configure_with_budget([1000].into_iter(), 0);
        assert!(matches!(scratch.diagonals[0], DiagonalTracker::Sparse(_)));
    }
}
