//! Utilities struct to initialize a gpu device.

use std::sync::Arc;
use wgpu::{Adapter, Backends, Device, Instance, InstanceDescriptor, Queue};

/// Helper struct to initialize a device and its queue.
pub struct GpuInstance {
    _instance: Instance, // TODO: do we have to keep this around?
    adapter: Adapter,    // TODO: do we have to keep this around?
    device: Arc<Device>,
    queue: Queue,
}

impl GpuInstance {
    pub async fn new() -> anyhow::Result<Self> {
        Self::with_backends(Backends::all()).await
    }

    pub async fn without_gl() -> anyhow::Result<Self> {
        Self::with_backends(Backends::all() & (!Backends::GL)).await
    }

    /// Initializes a wgpu instance and create its queue.
    pub async fn with_backends(backends: Backends) -> anyhow::Result<Self> {
        let instance_desc = InstanceDescriptor {
            backends,
            ..Default::default()
        };
        let instance = wgpu::Instance::new(&instance_desc);
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|_| anyhow::anyhow!("Failed to initialize gpu adapter."))?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::TIMESTAMP_QUERY,
                required_limits: wgpu::Limits {
                    max_buffer_size: 600_000_000,
                    max_storage_buffer_binding_size: 600_000_000,
                    ..Default::default()
                },
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| anyhow::anyhow!("{:?}", e))?;

        Ok(Self {
            _instance: instance,
            adapter,
            device: Arc::new(device),
            queue,
        })
    }

    /// The `wgpu` adapter.
    pub fn adapter(&self) -> &Adapter {
        &self.adapter
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
