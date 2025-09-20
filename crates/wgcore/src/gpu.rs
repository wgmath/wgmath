//! Utilities struct to initialize a gpu device.

use std::sync::Arc;
use wgpu::{Adapter, Device, Instance, Queue};

/// Helper struct to initialize a device and its queue.
pub struct GpuInstance {
    _instance: Instance, // TODO: do we have to keep this around?
    _adapter: Adapter,   // TODO: do we have to keep this around?
    device: Arc<Device>,
    queue: Queue,
}

impl GpuInstance {
    /// Initializes a wgpu instance and create its queue.
    pub async fn new() -> anyhow::Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .ok_or_else(|| anyhow::anyhow!("Failed to initialize gpu adapter."))?;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::TIMESTAMP_QUERY,
                    required_limits: wgpu::Limits {
                        max_buffer_size: 5_000_000_000,
                        max_storage_buffer_binding_size: 4_000_000_000,
                        ..Default::default()
                    },
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .map_err(|e| anyhow::anyhow!("{:?}", e))?;

        Ok(Self {
            _instance: instance,
            _adapter: adapter,
            device: Arc::new(device),
            queue,
        })
    }

    /// The `wgpu` device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The shared `wgpu` device.
    pub fn device_arc(&self) -> Arc<Device> {
        self.device.clone()
    }

    /// The `wgpu` queue.
    pub fn queue(&self) -> &Queue {
        &self.queue
    }
}
