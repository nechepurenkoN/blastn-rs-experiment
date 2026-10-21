//! Exclusive slot leases keep uploads, scratch and readback alive until collection.
use super::{gpu_error, pack::PackedBatch};
use blast_core::Result;
use std::{
    ops::{Deref, DerefMut},
    sync::{Condvar, Mutex},
};

#[derive(Default)]
pub(crate) struct Slot {
    pub packed: PackedBatch,
    pub buffers: Option<Buffers>,
}

pub(crate) struct Buffers {
    pub dna: wgpu::Buffer,
    pub jobs: wgpu::Buffer,
    pub scratch: wgpu::Buffer,
    pub output: wgpu::Buffer,
    pub staging: wgpu::Buffer,
    pub params: wgpu::Buffer,
    pub bindings: wgpu::BindGroup,
}

fn capacity(size: usize, limit: usize) -> u64 {
    size.max(4).next_power_of_two().min(limit) as u64
}

impl Slot {
    /// Grow buffers independently. Rebuild the bind group only when a handle changes.
    /// Returns (new buffer allocations, new bind groups).
    pub fn ensure_buffers(
        &mut self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        limit: usize,
    ) -> (usize, usize) {
        let p = &self.packed;
        let sizes = [
            p.dna.len(),
            p.jobs.len(),
            p.scratch_bytes,
            p.count * 48,
            p.count * 48,
            16,
        ];
        let previous = self.buffers.take();
        let old = previous.map(|b| {
            (
                [b.dna, b.jobs, b.scratch, b.output, b.staging, b.params],
                b.bindings,
            )
        });
        let (mut old_buffers, old_bindings) = match old {
            Some((buffers, bindings)) => (buffers.map(Some), Some(bindings)),
            None => ([const { None }; 6], None),
        };
        use wgpu::BufferUsages as U;
        let usages = [
            U::STORAGE | U::COPY_DST,
            U::STORAGE | U::COPY_DST,
            // Tests poison reusable DP storage to detect reads outside initialized intervals.
            U::STORAGE | if cfg!(test) { U::COPY_DST } else { U::empty() },
            U::STORAGE | U::COPY_SRC,
            U::COPY_DST | U::MAP_READ,
            U::UNIFORM | U::COPY_DST,
        ];
        let labels = [
            "DNA",
            "jobs",
            "DP rows",
            "scores",
            "readback",
            "active jobs",
        ];
        let mut allocations = 0;
        let buffers: [_; 6] = std::array::from_fn(|i| {
            if let Some(b) = old_buffers[i]
                .take()
                .filter(|b| b.size() >= sizes[i] as u64)
            {
                b
            } else {
                allocations += 1;
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(labels[i]),
                    size: capacity(sizes[i], limit),
                    usage: usages[i],
                    mapped_at_creation: false,
                })
            }
        });
        let [dna, jobs, scratch, output, staging, params] = buffers;
        let (bindings, groups) = if allocations == 0 {
            (old_bindings.unwrap(), 0)
        } else {
            let entries: Vec<_> = [&dna, &jobs, &scratch, &output, &params]
                .into_iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect();
            (
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("pooled extension"),
                    layout,
                    entries: &entries,
                }),
                1,
            )
        };
        self.buffers = Some(Buffers {
            dna,
            jobs,
            scratch,
            output,
            staging,
            params,
            bindings,
        });
        (allocations, groups)
    }
}

#[derive(Default)]
struct State {
    idle: Vec<Slot>,
    allocated: usize,
}

pub(crate) struct Pool {
    state: Mutex<State>,
    available: Condvar,
    max_slots: usize,
}

impl Pool {
    pub fn new(max_slots: usize) -> Self {
        Self {
            state: Mutex::new(State::default()),
            available: Condvar::new(),
            max_slots,
        }
    }

    pub fn checkout(&self, blocking: bool) -> Result<Option<Lease<'_>>> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| gpu_error("buffer pool poisoned"))?;
        loop {
            let slot = if let Some(slot) = state.idle.pop() {
                Some(slot)
            } else if state.allocated < self.max_slots {
                state.allocated += 1;
                Some(Slot::default())
            } else {
                None
            };
            if let Some(slot) = slot {
                return Ok(Some(Lease {
                    pool: self,
                    slot: Some(slot),
                    reusable: true,
                }));
            }
            if !blocking {
                return Ok(None);
            }
            state = self
                .available
                .wait(state)
                .map_err(|_| gpu_error("buffer pool poisoned"))?;
        }
    }
}

pub(crate) struct Lease<'a> {
    pool: &'a Pool,
    slot: Option<Slot>,
    /// Set false before submission; true only after a successful wait and unmap.
    pub reusable: bool,
}
impl Deref for Lease<'_> {
    type Target = Slot;
    fn deref(&self) -> &Slot {
        self.slot.as_ref().unwrap()
    }
}
impl DerefMut for Lease<'_> {
    fn deref_mut(&mut self) -> &mut Slot {
        self.slot.as_mut().unwrap()
    }
}
impl Drop for Lease<'_> {
    fn drop(&mut self) {
        let slot = self.slot.take().unwrap();
        if !self.reusable
            && let Some(buffers) = &slot.buffers
        {
            // Cancel pending mapping on error. Retire rather than recycle an in-flight slot.
            buffers.staging.unmap();
        }
        let mut state = self.pool.state.lock().unwrap_or_else(|e| e.into_inner());
        if self.reusable {
            state.idle.push(slot);
        } else {
            state.allocated -= 1;
        }
        self.pool.available.notify_one();
    }
}
