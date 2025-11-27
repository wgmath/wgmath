#![allow(clippy::result_large_err)]
#![allow(clippy::too_many_arguments)]

#[cfg(feature = "dim2")]
use rapier2d as rapier;
#[cfg(feature = "dim3")]
use rapier3d as rapier;
#[cfg(feature = "dim2")]
use wgrapier2d as wgrapier;
#[cfg(feature = "dim3")]
use wgrapier3d as wgrapier;

mod backend;
mod graphics;
mod ui;

use backend::{BackendType, CpuBackend, GpuBackend, PhysicsBackend};
use graphics::{setup_graphics, update_instances, RenderContext};
use ui::{render_compiling_message, render_ui, PhysicsContext, RunState};

#[cfg(feature = "dim2")]
use kiss3d::camera::FixedView;
use kiss3d::light::Light;
#[cfg(feature = "dim3")]
use kiss3d::prelude::PlanarFixedView;
#[cfg(feature = "dim2")]
use kiss3d::prelude::Sidescroll;
use kiss3d::window::Window;
use rapier::geometry::ColliderSet;
use rapier::prelude::{ImpulseJointSet, RigidBodySet};
use wgcore::gpu::GpuInstance;
use wgrapier::pipeline::{GpuPhysicsPipeline, RunStats};

pub struct SimulationState {
    pub bodies: RigidBodySet,
    pub colliders: ColliderSet,
    pub impulse_joints: ImpulseJointSet,
}

pub type SimulationBuilders = Vec<(&'static str, fn() -> SimulationState)>;

pub struct Testbed {
    builders: SimulationBuilders,
    selected_demo: usize,
    backend_type: BackendType,
    run_state: RunState,
    run_stats: RunStats,
    gpu_init_error: Option<String>,
    /// Cached GPU pipeline to avoid recompilation when switching demos
    cached_gpu_pipeline: Option<GpuPhysicsPipeline>,
}

impl Testbed {
    pub fn from_builders(builders: SimulationBuilders) -> Self {
        Self {
            builders,
            selected_demo: 0,
            backend_type: BackendType::Gpu { use_jacobi: false },
            run_state: RunState::Paused,
            run_stats: RunStats::default(),
            gpu_init_error: None,
            cached_gpu_pipeline: None,
        }
    }

    pub fn with_backend(mut self, backend_type: BackendType) -> Self {
        self.backend_type = backend_type;
        self
    }

    pub async fn run(mut self) {
        let mut window = Window::new("wgrapier demos");
        window.set_light(Light::StickToCamera);

        // Set up cameras first so we can render the "compiling" message
        #[cfg(feature = "dim2")]
        let (mut camera2d, mut camera3d) = {
            let mut sidescroll = Sidescroll::default();
            sidescroll.look_at(nalgebra::Point2::new(0.0, 100.0), 7.5);
            (sidescroll, FixedView::new())
        };
        #[cfg(feature = "dim3")]
        let (mut camera2d, mut camera3d) = {
            let arc_ball = kiss3d::prelude::ArcBall::new(
                nalgebra::Point3::new(-100.0, 100.0, -100.0),
                nalgebra::Point3::new(0.0, 40.0, 0.0),
            );
            (PlanarFixedView::new(), arc_ball)
        };

        // Try to initialize GPU, fallback to CPU if it fails
        let gpu = match GpuInstance::without_gl().await {
            Ok(gpu) => Some(gpu),
            Err(e) => {
                // GPU initialization failed, force CPU backend
                self.gpu_init_error = Some(format!(
                    "GPU backend not available, initialization failed:\n\"{}\"\n",
                    e
                ));
                self.backend_type = BackendType::Cpu;
                None
            }
        };

        // Check if we need to compile shaders (GPU backend without cached pipeline).
        let needs_shader_compilation = matches!(self.backend_type, BackendType::Gpu { .. })
            && self.cached_gpu_pipeline.is_none();

        // Render a "compiling shaders" message before doing the actual compilation.
        // The app will freeze during the compilation, so we need to draw this before.
        if needs_shader_compilation {
            // Don’t run a single render pass. It can take a few frames for the window/canvas to
            // show up so we don’t want the app to freeze before the message is actually visible.
            for _ in 0..100 {
                window
                    .render_with_cameras(&mut camera3d, &mut camera2d)
                    .await;
                render_compiling_message(&mut window);
            }
        }

        let phys = (self.builders[0].1)();
        let mut physics = setup_physics(
            gpu.as_ref(),
            &phys,
            self.backend_type,
            &mut self.gpu_init_error,
            &mut self.cached_gpu_pipeline,
        )
        .await;
        println!("There");

        let mut render_ctx = setup_graphics(&mut window, &phys).await;

        while window
            .render_with_cameras(&mut camera3d, &mut camera2d)
            .await
        {
            let ui_res = render_ui(
                &mut window,
                &self.builders,
                &mut self.selected_demo,
                &mut self.backend_type,
                &mut self.run_state,
                &self.run_stats,
                &mut physics,
                gpu.as_ref(),
                &self.gpu_init_error,
            );

            if let Some(new_demo) = ui_res.new_selected_demo {
                self.selected_demo = new_demo;
                let phys = (self.builders[new_demo].1)();
                render_ctx.clear();

                // Extract pipeline from current GPU backend if present
                if let PhysicsBackend::Gpu(gpu_backend) = physics.backend {
                    self.cached_gpu_pipeline = Some(gpu_backend.into_pipeline());
                }

                physics = setup_physics(
                    gpu.as_ref(),
                    &phys,
                    self.backend_type,
                    &mut self.gpu_init_error,
                    &mut self.cached_gpu_pipeline,
                )
                .await;

                render_ctx = setup_graphics(&mut window, &phys).await;
            }

            self.step_simulation(gpu.as_ref(), &mut physics, &mut render_ctx)
                .await;
        }
    }

    async fn step_simulation(
        &mut self,
        gpu: Option<&GpuInstance>,
        physics: &mut PhysicsContext,
        render_ctx: &mut RenderContext,
    ) {
        if self.run_state != RunState::Paused {
            self.run_stats = physics.backend.step(gpu, physics.timestamps.as_mut()).await;
        }

        if self.run_state == RunState::Step {
            self.run_state = RunState::Paused;
        }

        // Update instances using set_instances for efficient rendering
        update_instances(render_ctx, &physics.backend);
    }
}

async fn setup_physics(
    gpu: Option<&GpuInstance>,
    phys: &SimulationState,
    backend_type: BackendType,
    gpu_error: &mut Option<String>,
    cached_pipeline: &mut Option<GpuPhysicsPipeline>,
) -> PhysicsContext {
    let backend = match backend_type {
        BackendType::Gpu { use_jacobi } => {
            // Try to create GPU backend, fallback to CPU if it fails
            let gpu = gpu.unwrap();

            // Try to reuse cached pipeline or create a new one
            if let Some(pipeline) = cached_pipeline.take() {
                // Fast path: reuse existing pipeline
                let gpu_backend = GpuBackend::with_pipeline(gpu, pipeline, phys, use_jacobi).await;
                PhysicsBackend::Gpu(gpu_backend)
            } else {
                // Slow path: compile shaders for the first time
                match GpuBackend::try_new(gpu, phys, use_jacobi).await {
                    Ok(gpu_backend) => PhysicsBackend::Gpu(gpu_backend),
                    Err(e) => {
                        // GPU backend creation failed, fallback to CPU
                        *gpu_error = Some(format!(
                            "GPU backend initialization failed: {}. Using CPU backend.",
                            e
                        ));
                        PhysicsBackend::Cpu(CpuBackend::new(SimulationState {
                            bodies: phys.bodies.clone(),
                            colliders: phys.colliders.clone(),
                            impulse_joints: phys.impulse_joints.clone(),
                        }))
                    }
                }
            }
        }
        BackendType::Cpu => PhysicsBackend::Cpu(CpuBackend::new(SimulationState {
            bodies: phys.bodies.clone(),
            colliders: phys.colliders.clone(),
            impulse_joints: phys.impulse_joints.clone(),
        })),
    };

    PhysicsContext::new(backend)
}
