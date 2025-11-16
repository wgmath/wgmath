#[cfg(feature = "dim2")]
use wgrapier2d as wgrapier;
#[cfg(feature = "dim3")]
use wgrapier3d as wgrapier;

use super::SimulationBackend;
use crate::SimulationState;
use wgcore::gpu::GpuInstance;
use wgcore::tensor::GpuVector;
use wgcore::timestamps::GpuTimestamps;
use wgpu::BufferUsages;
use wgrapier::pipeline::{GpuPhysicsPipeline, GpuPhysicsState, RunStats};
use wgrapier::wgparry::math::GpuSim;

/// GPU-based physics backend using wgrapier
pub struct GpuBackend {
    pipeline: GpuPhysicsPipeline,
    state: GpuPhysicsState,
    poses_staging: GpuVector<GpuSim>,
    poses_cache: Vec<GpuSim>,
    use_jacobi: bool,
}

impl GpuBackend {
    /// Attempts to create a new GPU backend, returning an error if initialization fails.
    ///
    /// This method can fail if:
    /// - Shader compilation fails
    /// - GPU device doesn't support required features
    /// - Memory allocation fails
    pub async fn try_new(
        gpu: &GpuInstance,
        phys: &SimulationState,
        use_jacobi: bool,
    ) -> Result<Self, String> {
        let pipeline = GpuPhysicsPipeline::from_device(gpu.device())
            .map_err(|e| format!("Failed to compile shaders: {}", e))?;

        let state = GpuPhysicsState::from_rapier(
            gpu.device(),
            &phys.bodies,
            &phys.colliders,
            &phys.impulse_joints,
            use_jacobi,
        );
        let poses_staging = GpuVector::uninit(
            gpu.device(),
            state.poses().len() as u32,
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        );
        let poses_cache = state.poses().slow_read(gpu).await;

        Ok(Self {
            pipeline,
            state,
            poses_staging,
            poses_cache,
            use_jacobi,
        })
    }

    /// Creates a new GPU backend with a pre-compiled pipeline.
    ///
    /// This is faster than [`try_new`](Self::try_new) when switching demos because
    /// it reuses the existing pipeline instead of recompiling shaders.
    pub async fn with_pipeline(
        gpu: &GpuInstance,
        pipeline: GpuPhysicsPipeline,
        phys: &SimulationState,
        use_jacobi: bool,
    ) -> Self {
        let state = GpuPhysicsState::from_rapier(
            gpu.device(),
            &phys.bodies,
            &phys.colliders,
            &phys.impulse_joints,
            use_jacobi,
        );
        let poses_staging = GpuVector::uninit(
            gpu.device(),
            state.poses().len() as u32,
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        );
        let poses_cache = state.poses().slow_read(gpu).await;

        Self {
            pipeline,
            state,
            poses_staging,
            poses_cache,
            use_jacobi,
        }
    }

    /// Extracts the pipeline from this backend, consuming it.
    ///
    /// Useful for reusing the pipeline when switching demos.
    pub fn into_pipeline(self) -> GpuPhysicsPipeline {
        self.pipeline
    }

    /// Creates a new GPU backend, panicking if initialization fails.
    ///
    /// Use [`try_new`](Self::try_new) for error handling.
    #[allow(dead_code)]
    pub async fn new(gpu: &GpuInstance, phys: &SimulationState, use_jacobi: bool) -> Self {
        Self::try_new(gpu, phys, use_jacobi).await.unwrap()
    }
}

impl SimulationBackend for GpuBackend {
    fn poses(&self) -> &[GpuSim] {
        &self.poses_cache
    }
    fn num_bodies(&self) -> usize {
        self.poses_cache.len()
    }
    fn num_joints(&self) -> usize {
        self.state.joints().len()
    }

    async fn step(
        &mut self,
        gpu: Option<&GpuInstance>,
        mut timestamps: Option<&mut GpuTimestamps>,
    ) -> RunStats {
        let gpu = gpu.unwrap();
        if let Some(timestamps) = &mut timestamps {
            timestamps.clear();
        }

        let t0 = web_time::Instant::now();
        let mut run_stats = self
            .pipeline
            .step(
                gpu,
                &mut self.state,
                timestamps.as_deref_mut(),
                self.use_jacobi,
            )
            .await;

        // Read back poses and timestamps from GPU
        let mut encoder = gpu.device().create_command_encoder(&Default::default());
        if let Some(timestamps) = &mut timestamps {
            timestamps.resolve(&mut encoder);
        }

        self.poses_staging
            .copy_from(&mut encoder, self.state.poses());
        gpu.queue().submit(Some(encoder.finish()));

        self.poses_cache
            .resize(self.poses_staging.len() as usize, Default::default());
        self.poses_staging
            .read_to(gpu.device(), &mut self.poses_cache)
            .await
            .unwrap();
        run_stats.total_simulation_time_with_readback = t0.elapsed();

        let timestamps = if let Some(timestamps) = &mut timestamps {
            timestamps
                .wait_for_results_ms_async(gpu.queue(), gpu.device())
                .await
                .unwrap()
        } else {
            vec![]
        };

        if !timestamps.is_empty() {
            let timings = timestamps
                .chunks_exact(2)
                .map(|t| t[1] - t[0])
                .collect::<Vec<_>>();
            run_stats.timestamp_update_mass_props = timings[0];
            run_stats.timestamp_broad_phase = timings[1];
            run_stats.timestamp_narrow_phase = timings[2];
            run_stats.timestamp_solver_prep = timings[3];
            run_stats.timestamp_solver_solve = timings[4];
        }

        run_stats
    }
}
