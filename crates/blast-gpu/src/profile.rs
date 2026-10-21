use blast_core::align::{BATCH_HISTOGRAM_BINS, BatchDistribution, batch_histogram_bin};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

#[derive(Default)]
pub(super) struct Distribution {
    bins: [AtomicUsize; BATCH_HISTOGRAM_BINS],
    total: AtomicUsize,
    max: AtomicUsize,
}
impl Distribution {
    pub fn record(&self, size: usize) {
        self.bins[batch_histogram_bin(size)].fetch_add(1, Relaxed);
        self.total.fetch_add(size, Relaxed);
        self.max.fetch_max(size, Relaxed);
    }

    /// Aggregate a dispatch locally instead of contending on atomics per flank.
    pub fn record_many(&self, sizes: impl Iterator<Item = usize>) {
        let mut histogram = [0; BATCH_HISTOGRAM_BINS];
        let (mut total, mut max) = (0, 0);
        for size in sizes {
            histogram[batch_histogram_bin(size)] += 1;
            total += size;
            max = max.max(size);
        }
        for (bin, count) in self.bins.iter().zip(histogram) {
            if count != 0 {
                bin.fetch_add(count, Relaxed);
            }
        }
        self.total.fetch_add(total, Relaxed);
        self.max.fetch_max(max, Relaxed);
    }

    pub fn snapshot(&self) -> BatchDistribution {
        let histogram = self.bins.each_ref().map(|n| n.load(Relaxed));
        BatchDistribution {
            count: histogram.iter().sum(),
            total: self.total.load(Relaxed),
            max: self.max.load(Relaxed),
            histogram,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_boundaries_and_concurrent_totals() {
        let sizes = [
            0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512, 1023, 1024,
            2048,
        ];
        let distribution = Distribution::default();
        std::thread::scope(|scope| {
            for worker in 0..4 {
                let distribution = &distribution;
                scope.spawn(move || {
                    if worker % 2 == 0 {
                        for size in sizes {
                            distribution.record(size);
                        }
                    } else {
                        distribution.record_many(sizes.into_iter());
                    }
                });
            }
        });
        let stats = distribution.snapshot();
        assert_eq!(stats.count, 4 * sizes.len());
        assert_eq!(stats.total, 4 * sizes.iter().sum::<usize>());
        assert_eq!(stats.max, 2048);
        assert_eq!(stats.histogram, [4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]);
        assert_eq!(batch_histogram_bin(usize::MAX), 11);
    }
}
