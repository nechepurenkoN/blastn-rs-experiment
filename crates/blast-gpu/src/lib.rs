//! Optional portable GPU score-only backend.
//! Each invocation extends one seed flank; independent candidates run in parallel.
//! This is a correctness-first kernel, not a tiled/wavefront optimized SW kernel.
mod pack;
mod pool;
mod profile;

use blast_core::{
    Error, Result,
    align::{AlignOptions, AlignRequest, AlignmentScore, BatchProfile, GappedAligner},
};
use pool::{Lease, Pool};
use std::{
    collections::VecDeque,
    ops::Range,
    sync::{
        atomic::{AtomicUsize, Ordering::Relaxed},
        mpsc,
    },
    time::Duration,
};

/// Transport tuning only: neither option changes DP semantics or candidate selection.
#[derive(Clone, Copy, Debug)]
pub struct GpuOptions {
    pub batch_size: usize,
    pub in_flight: usize,
}
impl Default for GpuOptions {
    fn default() -> Self {
        Self {
            batch_size: 256,
            in_flight: 2,
        }
    }
}
impl GpuOptions {
    pub fn validate(self) -> Result<()> {
        if !(1..=1024).contains(&self.batch_size) || !(1..=8).contains(&self.in_flight) {
            return Err(gpu_error(
                "batch size must be 1..1024 and in-flight slots 1..8",
            ));
        }
        Ok(())
    }
}

/// Cumulative transport counters (not GPU hardware timing).
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuDiagnostics {
    pub submitted_batches: usize,
    pub buffer_allocations: usize,
    pub bind_groups_created: usize,
    pub bytes_uploaded: usize,
}
#[derive(Default)]
struct Counters {
    submitted_batches: AtomicUsize,
    buffer_allocations: AtomicUsize,
    bind_groups_created: AtomicUsize,
    bytes_uploaded: AtomicUsize,
    calls: profile::Distribution,
    dispatches: profile::Distribution,
    query_flanks: profile::Distribution,
    subject_bases: AtomicUsize,
}

pub struct GpuAligner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    limit: usize,
    options: GpuOptions,
    pool: Pool,
    counters: Counters,
    pub adapter_name: String,
}
struct Pending<'a> {
    lease: Lease<'a>,
    range: Range<usize>,
    submission: wgpu::SubmissionIndex,
    receiver: mpsc::Receiver<std::result::Result<(), wgpu::BufferAsyncError>>,
}
fn gpu_error(e: impl std::fmt::Display) -> Error {
    Error(format!("GPU: {e}"))
}

impl GpuAligner {
    pub fn new() -> Result<Self> {
        Self::with_options(GpuOptions::default())
    }

    pub fn with_options(options: GpuOptions) -> Result<Self> {
        options.validate()?;
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .map_err(gpu_error)?;
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu {
            return Err(gpu_error("only a software adapter is available"));
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("blast-rs"),
            ..Default::default()
        }))
        .map_err(gpu_error)?;
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("seeded affine extension"),
            source: wgpu::ShaderSource::Wgsl(include_str!("extend.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("seeded scores"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(e) = pollster::block_on(device.pop_error_scope()) {
            return Err(gpu_error(e));
        }
        let limit = device.limits().max_storage_buffer_binding_size as usize;
        let layout = pipeline.get_bind_group_layout(0);
        Ok(Self {
            device,
            queue,
            pipeline,
            layout,
            limit,
            options,
            pool: Pool::new(options.in_flight),
            counters: Counters::default(),
            adapter_name: format!("{} ({:?})", info.name, info.backend),
        })
    }

    pub fn diagnostics(&self) -> GpuDiagnostics {
        GpuDiagnostics {
            submitted_batches: self.counters.submitted_batches.load(Relaxed),
            buffer_allocations: self.counters.buffer_allocations.load(Relaxed),
            bind_groups_created: self.counters.bind_groups_created.load(Relaxed),
            bytes_uploaded: self.counters.bytes_uploaded.load(Relaxed),
        }
    }

    fn submit<'a>(&self, mut lease: Lease<'a>, range: Range<usize>) -> Pending<'a> {
        let (allocations, groups) = lease.ensure_buffers(&self.device, &self.layout, self.limit);
        self.counters
            .buffer_allocations
            .fetch_add(allocations, Relaxed);
        self.counters.bind_groups_created.fetch_add(groups, Relaxed);
        // Even queued uploads may be flushed by another caller's submission.
        lease.reusable = false;
        let p = &lease.packed;
        self.counters.dispatches.record(p.count);
        let mut subject_bases = 0;
        self.counters
            .query_flanks
            .record_many(p.jobs.chunks_exact(64).map(|job| {
                let slen = u32::from_le_bytes(job[12..16].try_into().unwrap()) as usize;
                subject_bases += slen;
                u32::from_le_bytes(job[4..8].try_into().unwrap()) as usize
            }));
        self.counters
            .subject_bases
            .fetch_add(subject_bases, Relaxed);
        let buffers = lease.buffers.as_ref().unwrap();
        self.queue.write_buffer(&buffers.dna, 0, &p.dna);
        self.queue.write_buffer(&buffers.jobs, 0, &p.jobs);
        let params: Vec<_> = [p.count as u32 * 2, 0, 0, 0]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        self.queue.write_buffer(&buffers.params, 0, &params);
        self.counters
            .bytes_uploaded
            .fetch_add(p.dna.len() + p.jobs.len() + 16, Relaxed);
        let output_size = (p.count * 48) as u64;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &buffers.bindings, &[]);
            pass.dispatch_workgroups((p.count * 2).div_ceil(32) as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&buffers.output, 0, &buffers.staging, 0, output_size);
        let submission = self.queue.submit([encoder.finish()]);
        self.counters.submitted_batches.fetch_add(1, Relaxed);
        let (sender, receiver) = mpsc::channel();
        lease
            .buffers
            .as_ref()
            .unwrap()
            .staging
            .slice(..output_size)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = sender.send(r);
            });
        Pending {
            lease,
            range,
            submission,
            receiver,
        }
    }

    fn collect(
        &self,
        mut pending: Pending<'_>,
        requests: &[AlignRequest<'_>],
        options: AlignOptions,
        result: &mut Vec<AlignmentScore>,
    ) -> Result<()> {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(pending.submission),
                timeout: Some(Duration::from_secs(60)),
            })
            .map_err(gpu_error)?;
        pending
            .receiver
            .recv_timeout(Duration::from_secs(1))
            .map_err(gpu_error)?
            .map_err(gpu_error)?;
        let staging = &pending.lease.buffers.as_ref().unwrap().staging;
        let mapped = staging
            .slice(..(pending.range.len() * 48) as u64)
            .get_mapped_range();
        let mut exceeded = false;
        for (&r, bytes) in requests[pending.range].iter().zip(mapped.chunks_exact(48)) {
            let words: [u32; 12] = std::array::from_fn(|i| {
                u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap())
            });
            let (left, right) = words.split_at(6);
            if left[4] != 0 || right[4] != 0 {
                exceeded = true;
                break;
            }
            result.push(AlignmentScore {
                score: left[0] as i32
                    + right[0] as i32
                    + r.word_size as i32 * options.scoring.reward,
                query_start: r.query_anchor - left[1] as usize,
                query_end: r.query_anchor + r.word_size + right[1] as usize,
                subject_start: r.subject_anchor - left[2] as usize,
                subject_end: r.subject_anchor + r.word_size + right[2] as usize,
                band_edge_reached: left[3] != 0 || right[3] != 0,
            });
        }
        drop(mapped);
        staging.unmap();
        pending.lease.reusable = true;
        if exceeded {
            return Err(gpu_error("alignment cell budget exceeded"));
        }
        Ok(())
    }
}

impl GappedAligner for GpuAligner {
    fn name(&self) -> &'static str {
        "gpu"
    }

    fn shared_query_batch_size(&self) -> Option<usize> {
        Some(self.options.batch_size)
    }

    fn batch_profile(&self) -> Option<BatchProfile> {
        let transport = self.diagnostics();
        Some(BatchProfile {
            backend: "gpu",
            calls: self.counters.calls.snapshot(),
            dispatches: self.counters.dispatches.snapshot(),
            query_flanks: self.counters.query_flanks.snapshot(),
            subject_bases: self.counters.subject_bases.load(Relaxed),
            bytes_uploaded: transport.bytes_uploaded,
            buffer_allocations: transport.buffer_allocations,
            bind_groups_created: transport.bind_groups_created,
        })
    }

    fn score_batch(
        &self,
        requests: &[AlignRequest<'_>],
        options: AlignOptions,
    ) -> Result<Vec<AlignmentScore>> {
        options.validate()?;
        self.counters.calls.record(requests.len());
        let mut pending = VecDeque::new();
        let mut result = Vec::with_capacity(requests.len());
        let mut next = 0;
        while next < requests.len() || !pending.is_empty() {
            if next < requests.len() {
                // Never wait for a slot while owning outstanding work: concurrent callers
                // could otherwise exhaust the pool and deadlock one another.
                if let Some(mut lease) = self.pool.checkout(pending.is_empty())? {
                    let end = requests.len().min(next + self.options.batch_size);
                    if let Err(error) =
                        lease
                            .packed
                            .prepare(&requests[next..end], options, self.limit)
                    {
                        // Report earlier GPU errors before a later packing error.
                        drop(lease);
                        while let Some(batch) = pending.pop_front() {
                            self.collect(batch, requests, options, &mut result)?;
                        }
                        return Err(error);
                    }
                    let end = next + lease.packed.count;
                    pending.push_back(self.submit(lease, next..end));
                    next = end;
                    continue;
                }
            }
            if let Some(batch) = pending.pop_front() {
                self.collect(batch, requests, options, &mut result)?;
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use blast_core::align::{ScoringScheme, align_seeded};

    // A single-slot test queues this upload before the next dispatch, including
    // regions outside the previous live interval. Fresh/zeroed storage could hide
    // missing initialization or stale reads after a frontier or layout change.
    fn poison_scratch(gpu: &GpuAligner, byte: u8) {
        let lease = gpu.pool.checkout(true).unwrap().unwrap();
        if let Some(buffers) = &lease.buffers {
            gpu.queue.write_buffer(
                &buffers.scratch,
                0,
                &vec![byte; buffers.scratch.size() as usize],
            );
        }
    }

    #[test]
    #[ignore = "requires a physical Metal/Vulkan/DX12 GPU; run explicitly"]
    fn active_frontiers_match_cpu_with_dirty_reused_scratch_and_budget_boundaries() {
        let gpu = GpuAligner::with_options(GpuOptions {
            batch_size: 128,
            in_flight: 1,
        })
        .unwrap();
        let mut random = 771u64;
        let mut next = |n: usize| {
            random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
            (random >> 32) as usize % n
        };
        let mut data = vec![(vec![0; 512], vec![0; 512], 256, 256, 7)];
        for case in 1..64 {
            let k = 1 + next(32);
            let qlen = k + if case % 9 == 0 { 0 } else { next(180) };
            let slen = k + if case % 11 == 0 { 0 } else { next(220) };
            let mut query: Vec<_> = (0..qlen).map(|_| next(15) as u8).collect();
            let mut subject: Vec<_> = (0..slen).map(|_| next(4) as u8).collect();
            if case % 3 != 0 {
                for b in &mut query {
                    *b %= 4;
                }
                subject[..qlen.min(slen)].copy_from_slice(&query[..qlen.min(slen)]);
                let gap = (1 + next(49)).min(slen / 2);
                subject.splice(slen / 2..slen / 2, std::iter::repeat_n(14, gap));
            }
            let qa = match case % 4 {
                0 => 0,
                1 => qlen - k,
                _ => next(qlen - k + 1),
            };
            let sa = match case % 4 {
                0 => 0,
                1 => subject.len() - k,
                _ => next(subject.len() - k + 1),
            };
            for i in 0..k {
                query[qa + i] = next(4) as u8;
                subject[sa + i] = query[qa + i];
            }
            data.push((query, subject, qa, sa, k));
        }
        let mut requests: Vec<_> = data
            .iter()
            .map(|(q, s, qa, sa, k)| AlignRequest {
                query: q,
                subject: s,
                query_anchor: *qa,
                subject_anchor: *sa,
                word_size: *k,
            })
            .collect();
        let mut iteration = 0;
        for band in [None, Some(0), Some(1), Some(7), Some(64)] {
            for xdrop in [1_000_000, 0, 1, 7, 50] {
                for scoring in [
                    ScoringScheme::default(),
                    ScoringScheme {
                        reward: 1,
                        penalty: -1,
                        gap_open: 0,
                        gap_extend: 1,
                    },
                    ScoringScheme {
                        reward: 100,
                        penalty: -100,
                        gap_open: 100,
                        gap_extend: 100,
                    },
                ] {
                    let options = AlignOptions {
                        band,
                        xdrop,
                        scoring,
                        ..Default::default()
                    };
                    poison_scratch(&gpu, [0x3f, 0, 0xc0, 0x7f][iteration % 4]);
                    let expected: Vec<_> = requests
                        .iter()
                        .map(|&r| align_seeded(r, options, false).unwrap().summary)
                        .collect();
                    let actual = gpu.score_batch(&requests, options).unwrap();
                    assert_eq!(
                        actual, expected,
                        "iteration={iteration} options={options:?}"
                    );
                    // Different job/row offsets on every call, using the same dirty buffer.
                    requests.rotate_left(7);
                    iteration += 1;
                }
            }
        }

        // Verify every small budget boundary, including the last dead row, without
        // batching errors hiding later jobs. GPU keeps its existing error wording.
        let query = [0; 12];
        for mut subject in [[0; 16], [1; 16]] {
            subject[0] = 0;
            let r = AlignRequest {
                query: &query,
                subject: &subject,
                query_anchor: 0,
                subject_anchor: 0,
                word_size: 1,
            };
            for band in [None, Some(0), Some(1)] {
                for max_cells in 1..=177 {
                    let options = AlignOptions {
                        band,
                        xdrop: 0,
                        max_cells,
                        ..Default::default()
                    };
                    poison_scratch(&gpu, 0x3f);
                    let actual = gpu.score_batch(&[r], options);
                    match align_seeded(r, options, false) {
                        Ok(expected) => assert_eq!(actual.unwrap(), [expected.summary]),
                        Err(error) => {
                            assert!(error.0.contains("cell budget exceeded"));
                            assert_eq!(
                                actual.unwrap_err().0,
                                "GPU: alignment cell budget exceeded"
                            );
                        }
                    }
                }
            }
        }
        // Recover after errors, change batch length/offsets, and exercise the u32 budget limit.
        let options = AlignOptions {
            max_cells: u32::MAX as usize,
            ..Default::default()
        };
        for count in [1, 33, 0, 64, 17] {
            poison_scratch(&gpu, 0x7f);
            let expected: Vec<_> = requests[..count]
                .iter()
                .map(|&r| align_seeded(r, options, false).unwrap().summary)
                .collect();
            assert_eq!(
                gpu.score_batch(&requests[..count], options).unwrap(),
                expected
            );
        }
        let uniform = [AlignRequest {
            query: &data[0].0,
            subject: &data[0].1,
            query_anchor: 256,
            subject_anchor: 256,
            word_size: 7,
        }];
        let scores = gpu
            .score_batch(
                &uniform,
                AlignOptions {
                    xdrop: 0,
                    scoring: ScoringScheme {
                        reward: 100,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(scores[0].score, 51_200);
    }

    #[test]
    fn invalid_transport_settings_are_rejected_without_a_device() {
        for options in [
            GpuOptions {
                batch_size: 0,
                in_flight: 2,
            },
            GpuOptions {
                batch_size: 1025,
                in_flight: 2,
            },
            GpuOptions {
                batch_size: 64,
                in_flight: 0,
            },
            GpuOptions {
                batch_size: 64,
                in_flight: 9,
            },
        ] {
            assert!(GpuAligner::with_options(options).is_err());
        }
    }

    #[test]
    #[ignore = "requires a physical Metal/Vulkan/DX12 GPU; run explicitly"]
    fn pooled_pipeline_reuse_concurrency_and_error_recovery() {
        let gpu = GpuAligner::with_options(GpuOptions {
            batch_size: 32,
            in_flight: 2,
        })
        .unwrap();
        let dna: Vec<_> = (0..128).map(|i| (i % 4) as u8).collect();
        let requests: Vec<_> = (0..96)
            .map(|i| AlignRequest {
                // Distinct scores/endpoints make reordered results observable.
                query: &dna[..96 + i % 33],
                subject: &dna[..96 + i % 33],
                query_anchor: i % 32,
                subject_anchor: i % 32,
                word_size: 11,
            })
            .collect();
        let options = AlignOptions::default();
        let expected: Vec<_> = requests
            .iter()
            .map(|&r| align_seeded(r, options, false).unwrap().summary)
            .collect();
        assert_eq!(gpu.score_batch(&requests, options).unwrap(), expected);
        let warm = gpu.diagnostics();
        assert_eq!(warm.submitted_batches, 3);
        assert_eq!(warm.buffer_allocations, 12);
        assert_eq!(warm.bind_groups_created, 2);
        let profile = gpu.batch_profile().unwrap();
        assert_eq!(profile.calls.count, 1);
        assert_eq!(profile.calls.total, 96);
        assert_eq!(profile.calls.histogram[7], 1);
        assert_eq!(profile.dispatches.count, 3);
        assert_eq!(profile.dispatches.total, 96);
        assert_eq!(profile.dispatches.histogram[6], 3);
        assert_eq!(profile.query_flanks.count, 192);
        let bases: usize = requests.iter().map(|r| r.query.len() - r.word_size).sum();
        assert_eq!(profile.query_flanks.total, bases);
        assert_eq!(profile.subject_bases, bases);
        assert_eq!(profile.bytes_uploaded, 8 * bases + 96 * 128 + 3 * 16);
        assert_eq!(profile.bytes_uploaded, warm.bytes_uploaded);
        // Short partial workgroups following full batches must not execute stale jobs.
        for n in [1, 17, 65, 0, 96] {
            assert_eq!(
                gpu.score_batch(&requests[..n], options).unwrap(),
                expected[..n]
            );
        }
        std::thread::scope(|scope| {
            for worker in 0..6 {
                let (gpu, requests, expected) = (&gpu, &requests, &expected);
                scope.spawn(move || {
                    for n in [96, 1 + worker, 65, 17] {
                        assert_eq!(
                            gpu.score_batch(&requests[..n], options).unwrap(),
                            expected[..n]
                        );
                    }
                });
            }
        });
        let reused = gpu.diagnostics();
        assert_eq!(reused.buffer_allocations, warm.buffer_allocations);
        assert_eq!(reused.bind_groups_created, warm.bind_groups_created);
        assert!(reused.submitted_batches > warm.submitted_batches);
        assert!(
            gpu.score_batch(
                &requests,
                AlignOptions {
                    max_cells: 1,
                    ..options
                }
            )
            .is_err()
        );
        assert_eq!(gpu.score_batch(&requests, options).unwrap(), expected);
        let mut invalid = requests.clone();
        invalid[70].query_anchor = usize::MAX;
        assert!(gpu.score_batch(&invalid, options).is_err());
        assert_eq!(gpu.score_batch(&requests, options).unwrap(), expected);
        assert!(
            gpu.score_batch(
                &[],
                AlignOptions {
                    max_cells: 0,
                    ..options
                }
            )
            .is_err()
        );
        let seed_only = AlignRequest {
            query: &dna[..11],
            subject: &dna[..11],
            query_anchor: 0,
            subject_anchor: 0,
            word_size: 11,
        };
        assert_eq!(
            gpu.score_batch(&[seed_only], options).unwrap(),
            [align_seeded(seed_only, options, false).unwrap().summary]
        );
        // Force budget-based splitting independently of the configured batch length.
        let mut single_slot = GpuAligner::with_options(GpuOptions {
            batch_size: 96,
            in_flight: 1,
        })
        .unwrap();
        single_slot.limit = 2500;
        assert_eq!(
            single_slot.score_batch(&requests, options).unwrap(),
            expected
        );
        assert_eq!(single_slot.diagnostics().submitted_batches, 96);
        single_slot.limit = 1000;
        assert!(single_slot.score_batch(&requests, options).is_err());
        single_slot.limit = 2500;
        assert_eq!(
            single_slot.score_batch(&requests[..1], options).unwrap(),
            expected[..1]
        );
    }
    #[test]
    #[ignore = "requires a physical Metal/Vulkan/DX12 GPU; run explicitly"]
    fn gpu_matches_cpu() {
        let gpu = GpuAligner::new().unwrap();
        eprintln!("GPU differential test: {}", gpu.adapter_name);
        let mut random = 29u64;
        let mut data = Vec::new();
        for i in 0..80 {
            let mut q: Vec<u8> = (0..i + 12)
                .map(|_| {
                    random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
                    (random >> 32) as u8 % 4
                })
                .collect();
            let mut s = q.clone();
            if i % 3 == 0 {
                s.splice(10..10, [3, 2, 1]);
            }
            if i % 5 == 0 {
                q[9] = 14;
            }
            if i % 7 == 0 {
                s[8] = (s[8] + 1) % 4;
            }
            data.push((q, s));
        }
        let requests: Vec<_> = data
            .iter()
            .enumerate()
            .map(|(i, (q, s))| AlignRequest {
                query: q,
                subject: s,
                query_anchor: if i % 2 == 0 { 0 } else { 2 },
                subject_anchor: if i % 2 == 0 { 0 } else { 2 },
                word_size: 5,
            })
            .collect();
        for band in [None, Some(2), Some(16)] {
            for xdrop in [3, 50] {
                let options = AlignOptions {
                    band,
                    xdrop,
                    ..Default::default()
                };
                let actual = gpu.score_batch(&requests, options).unwrap();
                for (i, (&r, a)) in requests.iter().zip(actual).enumerate() {
                    assert_eq!(
                        a,
                        align_seeded(r, options, false).unwrap().summary,
                        "case {i}, band {band:?}, drop {xdrop}"
                    );
                }
            }
        }
        assert!(
            gpu.score_batch(
                &requests,
                AlignOptions {
                    max_cells: 1,
                    ..Default::default()
                }
            )
            .is_err()
        );
    }
}
