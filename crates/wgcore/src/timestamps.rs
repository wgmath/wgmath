//! A convenient wrapper for handling gpu timestamps.
//!
//! Note that this is strongly inspired from wgpu’s timestamp queries example:
//! <https://github.com/gfx-rs/wgpu/blob/trunk/examples/src/timestamp_queries/mod.rs>

use wgpu::{BufferAsyncError, ComputePass, ComputePassTimestampWrites, Device, QuerySet, Queue};

/// A set of gpu timestamps, generally useful to determine shader execution times.
pub struct GpuTimestamps {
    set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    destination_buffer: wgpu::Buffer,
    capacity: u32,
    len: u32,
}

impl GpuTimestamps {
    /// Creates a set of gpu timestamps that has room for at most `capacity` timestamps.
    pub fn new(device: &wgpu::Device, capacity: u32) -> Self {
        GpuTimestamps {
            set: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("timestamps queries"),
                count: capacity,
                ty: wgpu::QueryType::Timestamp,
            }),
            resolve_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timestamps resolve buffer"),
                size: std::mem::size_of::<u64>() as u64 * capacity as u64,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
                mapped_at_creation: false,
            }),
            destination_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timestamps dest buffer"),
                size: std::mem::size_of::<u64>() as u64 * capacity as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            capacity,
            len: 0,
        }
    }

    /// Returns `true` if no timestamps was registered in this set.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// How many timestamps are registered in this set.
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// The underlying wgpu `QuerySet`.
    pub fn query_set(&self) -> &QuerySet {
        &self.set
    }

    /// Reserves the next two timestamp slots from this set and returns the corresponding
    /// `ComputePassTimestampWrites` descriptor to be given to a compute pass creation to measure
    /// its execution times.
    ///
    /// Returns `None` if there is no room form two additional timestamps in `self`.
    pub fn next_compute_pass_timestamp_writes(&mut self) -> Option<ComputePassTimestampWrites<'_>> {
        let ids = self.next_query_indices::<2>()?;
        Some(wgpu::ComputePassTimestampWrites {
            query_set: &self.set,
            beginning_of_pass_write_index: Some(ids[0]),
            end_of_pass_write_index: Some(ids[1]),
        })
    }

    /// Allocate a single timestamp into this set and return its index.
    ///
    /// Returns `None` if adding one timestamp would exceed this set’s capacity.
    pub fn next_query_index(&mut self) -> Option<u32> {
        self.next_query_indices::<1>().map(|idx| idx[0])
    }

    /// Allocate `COUNT` timestamp to this set and return their indices.
    ///
    /// Returns `None` if adding `COUNT` timestamp would exceed this set’s capacity.
    pub fn next_query_indices<const COUNT: usize>(&mut self) -> Option<[u32; COUNT]> {
        if COUNT == 0 {
            return Some([0; COUNT]);
        }

        if self.len + (COUNT as u32) - 1 < self.capacity {
            Some([0; COUNT].map(|_| {
                self.len += 1;
                self.len - 1
            }))
        } else {
            None
        }
    }

    /// Allocate a single timestamp into this set, and write it into the given `compute_pass`
    /// with [`ComputePass::write_timestamp`].
    pub fn write_next_timestamp(&mut self, compute_pass: &mut ComputePass) -> Option<u32> {
        let id = self.next_query_index()?;
        compute_pass.write_timestamp(&self.set, id);
        Some(id)
    }

    /// Writes the timestamp identified by `query_index` into the `compute_pass` using
    /// [`ComputePass::write_timestamp`]. It is assumed that the `query_index` has already
    /// been allocated into this set, such that `query_index < self.len()`.
    pub fn write_timestamp_at(&mut self, compute_pass: &mut ComputePass, query_index: u32) -> bool {
        if query_index < self.capacity {
            compute_pass.write_timestamp(&self.set, query_index);
            true
        } else {
            false
        }
    }

    /// Appends to the `encoder` commands to resolve the underlying query set and to retrieve the
    /// timestamp information from the gpu.
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.resolve_query_set(
            &self.set,
            // TODO(https://github.com/gfx-rs/wgpu/issues/3993): Musn't be larger than the number valid queries in the set.
            0..self.len,
            &self.resolve_buffer,
            0,
        );
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.destination_buffer,
            0,
            self.resolve_buffer.size(),
        );
    }

    /// Wait for the timestamps to be readable as a CPU/RAM buffer and return their raw (integer)
    /// values.
    ///
    /// Because this method is async, it is more suitable than `GpuTimestamps::wait_for_results`
    /// to be called from an async context, or when targeting web platforms.
    ///
    /// Note that the result is given as a vector or raw integer timestamps. To convert them
    /// into actual time measurements they need be multiplied by `Queue::get_timestamp_period`. See
    /// [`GpuTimestamps::wait_for_results_ms_async`] for a method that applies that multiplication
    /// automatically.
    pub async fn wait_for_results_async(
        &self,
        _device: &Device,
    ) -> Result<Vec<u64>, BufferAsyncError> {
        let (snd, rcv) = async_channel::bounded(1);
        self.destination_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    snd.send_blocking(r).unwrap();
                }
                #[cfg(target_arch = "wasm32")]
                {
                    let _ = snd.force_send(r).unwrap();
                }
            });
        #[cfg(not(target_arch = "wasm32"))]
        let _ = _device.poll(wgpu::PollType::wait());
        rcv.recv().await.unwrap()?;

        let timestamps = {
            let timestamp_view = self
                .destination_buffer
                .slice(
                    ..(std::mem::size_of::<u64>() as wgpu::BufferAddress
                        * self.capacity as wgpu::BufferAddress),
                )
                .get_mapped_range();
            bytemuck::cast_slice(&timestamp_view).to_vec()
        };

        self.destination_buffer.unmap();

        Ok(timestamps)
    }

    /// Wait for the timestamps to be readable as a CPU/RAM buffer and return their values in
    /// milliseconds.
    ///
    /// Because this method is async, it is more suitable than `GpuTimestamps::wait_for_results`
    /// to be called from an async context, or when targeting web platforms.
    pub async fn wait_for_results_ms_async(
        &self,
        queue: &Queue,
        device: &Device,
    ) -> Result<Vec<f64>, BufferAsyncError> {
        let timestamps = self.wait_for_results_async(device).await?;
        let period = queue.get_timestamp_period();
        Ok(Self::timestamps_to_ms(&timestamps, period))
    }

    /// The blocking counterpart of [`GpuTimestamps::wait_for_results_async`].
    ///
    /// This is not compatible with web platforms.
    pub fn wait_for_results(&self, device: &wgpu::Device) -> Vec<u64> {
        self.destination_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| ());
        let _ = device.poll(wgpu::PollType::wait());

        let timestamps = {
            let timestamp_view = self
                .destination_buffer
                .slice(
                    ..(std::mem::size_of::<u64>() as wgpu::BufferAddress
                        * self.capacity as wgpu::BufferAddress),
                )
                .get_mapped_range();
            bytemuck::cast_slice(&timestamp_view).to_vec()
        };

        self.destination_buffer.unmap();

        timestamps
    }

    /// The blocking counterpart of [`GpuTimestamps::wait_for_results_ms`].
    ///
    /// This is not compatible with web platforms.
    pub fn wait_for_results_ms(&self, device: &Device, queue: &Queue) -> Vec<f64> {
        let timestamps = self.wait_for_results(device);
        let period = queue.get_timestamp_period();
        Self::timestamps_to_ms(&timestamps, period)
    }

    /// Converts a set of raw timestamps into milliseconds.
    ///
    /// The `timestamp_period` should be the result of a call to [`Queue::get_timestamp_period`].
    pub fn timestamps_to_ms(timestamps: &[u64], timestamp_period: f32) -> Vec<f64> {
        timestamps
            .iter()
            .map(|t| *t as f64 * timestamp_period as f64 / 1_000_000.0)
            .collect()
    }

    /// Clears this set of timestamp.
    ///
    /// This sets the logical length to zero but the capacity/gpu buffer sizes are not modified.
    pub fn clear(&mut self) {
        self.len = 0;
    }
}
