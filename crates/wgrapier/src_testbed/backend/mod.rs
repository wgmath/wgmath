mod cpu;
mod gpu;

pub use cpu::CpuBackend;
pub use gpu::GpuBackend;

#[cfg(feature = "dim2")]
use wgrapier2d as wgrapier;
#[cfg(feature = "dim3")]
use wgrapier3d as wgrapier;

use wgcore::gpu::GpuInstance;
use wgcore::timestamps::GpuTimestamps;
use wgrapier::pipeline::RunStats;
use wgrapier::wgparry::math::GpuSim;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BackendType {
    Cpu,
    Gpu { use_jacobi: bool },
}

/// Trait for physics simulation backends (CPU or GPU)
pub trait SimulationBackend {
    /// Get the current poses for rendering
    fn poses(&self) -> &[GpuSim];

    /// Step the simulation
    async fn step(
        &mut self,
        gpu: Option<&GpuInstance>,
        timestamps: Option<&mut GpuTimestamps>,
    ) -> RunStats;
}

#[allow(clippy::large_enum_variant)]
pub enum PhysicsBackend {
    Cpu(CpuBackend),
    Gpu(GpuBackend),
}

impl PhysicsBackend {
    pub async fn step(
        &mut self,
        gpu: Option<&GpuInstance>,
        timestamps: Option<&mut GpuTimestamps>,
    ) -> RunStats {
        match self {
            PhysicsBackend::Cpu(backend) => backend.step(gpu, timestamps).await,
            PhysicsBackend::Gpu(backend) => backend.step(gpu, timestamps).await,
        }
    }

    pub fn poses(&self) -> &[GpuSim] {
        match self {
            PhysicsBackend::Cpu(backend) => backend.poses(),
            PhysicsBackend::Gpu(backend) => backend.poses(),
        }
    }
}
