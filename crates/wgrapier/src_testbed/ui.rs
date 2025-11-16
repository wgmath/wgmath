#[cfg(feature = "dim2")]
use wgrapier2d as wgrapier;
#[cfg(feature = "dim3")]
use wgrapier3d as wgrapier;

use crate::backend::{BackendType, PhysicsBackend};
use crate::SimulationBuilders;
use kiss3d::egui::CollapsingHeader;
use kiss3d::window::Window;
use wgcore::gpu::GpuInstance;
use wgcore::timestamps::GpuTimestamps;
use wgrapier::pipeline::RunStats;

const TIMESTAMP_QUERIES_CAPACITY: u32 = 20;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RunState {
    Running,
    Paused,
    Step,
}

#[derive(Default, Copy, Clone)]
pub struct UiInteractions {
    pub new_selected_demo: Option<usize>,
}

pub struct PhysicsContext {
    pub backend: PhysicsBackend,
    pub timestamps: Option<GpuTimestamps>,
}

impl PhysicsContext {
    pub fn new(backend: PhysicsBackend) -> Self {
        Self {
            backend,
            timestamps: None,
        }
    }
}

pub fn render_ui(
    window: &mut Window,
    builders: &SimulationBuilders,
    selected_demo: &mut usize,
    backend_type: &mut BackendType,
    run_state: &mut RunState,
    run_stats: &RunStats,
    physics: &mut PhysicsContext,
    gpu: Option<&GpuInstance>,
    gpu_init_error: &Option<String>,
) -> UiInteractions {
    let mut result = UiInteractions::default();

    window.draw_ui(|ctx| {
        kiss3d::egui::Window::new("Settings").show(ctx, |ui| {
            // Display GPU error message if present
            if let Some(error_msg) = gpu_init_error {
                ui.colored_label(
                    kiss3d::egui::Color32::from_rgb(255, 50, 50),
                    format!("⚠ {}", error_msg),
                );
                ui.separator();
            }

            if matches!(backend_type, BackendType::Gpu { .. }) && run_stats.total_simulation_time_with_readback.as_secs_f32() > 0.1 {
                ui.colored_label(
                    kiss3d::egui::Color32::from_rgb(255, 165, 0),
                    #[cfg(not(target_arch = "wasm32"))]
                    format!("⚠ running slow? If you have both an integrated and discrete GPU, ensure the discrete GPU is in use.\nSelected GPU: \"{}\"",
                            gpu.as_ref().unwrap().adapter().get_info().name),
                    #[cfg(target_arch = "wasm32")]
                    format!("⚠ running slow? If you have both an integrated and discrete GPU, ensure your browser runs exclusively on the discrete GPU."),
                );
                ui.separator();
            }

            let mut changed = false;
            kiss3d::egui::ComboBox::from_label("selected sample")
                .selected_text(builders[*selected_demo].0)
                .show_ui(ui, |ui| {
                    for (i, (name, _)) in builders.iter().enumerate() {
                        changed = ui.selectable_value(selected_demo, i, *name).changed() || changed;
                    }
                });
            if changed {
                result.new_selected_demo = Some(*selected_demo);
            }

            ui.separator();

            // Backend selection
            ui.label("Physics Backend:");
            ui.horizontal(|ui| {
                let mut backend_changed = false;
                if gpu.is_some()
                    && ui
                        .radio(
                            matches!(*backend_type, BackendType::Gpu { .. }),
                            "GPU (wgrapier)",
                        )
                        .clicked()
                    && !matches!(*backend_type, BackendType::Gpu { .. })
                {
                    *backend_type = BackendType::Gpu { use_jacobi: false };
                    backend_changed = true;
                }

                if ui
                    .radio(*backend_type == BackendType::Cpu, "CPU (rapier)")
                    .clicked()
                    && *backend_type != BackendType::Cpu
                {
                    *backend_type = BackendType::Cpu;
                    backend_changed = true;
                }
                if backend_changed {
                    // Reset the scene.
                    result.new_selected_demo = Some(*selected_demo);
                }
            });

            // NOTE: The jacobi solver is pretty much useless for the 2D version so we
            //       don’t even let the user enable it.
            #[cfg(feature = "dim3")]
            if let BackendType::Gpu { use_jacobi } = backend_type {
                if ui
                    .checkbox(use_jacobi, "Jacobi solver (more perf, less stable)")
                    .changed()
                {
                    // Reset the simulation with the new solver.
                    result.new_selected_demo = Some(*selected_demo);
                }
            }

            ui.separator();

            ui.label(format!("Bodies count: {}", physics.backend.num_bodies()));
            ui.label(format!("Joints count: {}", physics.backend.num_joints()));

            if *backend_type == BackendType::Cpu {
                ui.label(format!(
                    "Total: {:.2}ms - {} fps",
                    run_stats.total_simulation_time_ms(),
                    (1000.0f32 / run_stats.total_simulation_time_ms()).round()
                ));
            } else if matches!(*backend_type, BackendType::Gpu { .. }) {
                let mut timestamps_enabled = physics.timestamps.is_some();
                if ui
                    .checkbox(&mut timestamps_enabled, "enable timestamp queries")
                    .changed()
                {
                    if timestamps_enabled {
                        let gpu = gpu.unwrap();
                        physics.timestamps =
                            Some(GpuTimestamps::new(gpu.device(), TIMESTAMP_QUERIES_CAPACITY));
                    } else {
                        physics.timestamps = None;
                    }
                }

                if let Some(gpu) = gpu {
                    ui.collapsing("GPU infos", |ui| {
                        ui.label(format!("{:#?}", gpu.adapter().get_info()));
                    });
                }

                CollapsingHeader::new(format!(
                    "Total phys. runtime: {:.2}ms - {} fps",
                    run_stats.total_simulation_time_ms(),
                    (1000.0f32 / run_stats.total_simulation_time_ms()).round()
                ))
                .id_salt("total")
                .show(ui, |ui| {
                    ui.label(format!("num_colors: {}", run_stats.num_colors));
                    ui.label(format!(
                        "constraints_coloring_time: {:.2}",
                        run_stats.coloring_time.as_secs_f32() * 1000.0
                    ));
                    ui.label(format!(
                        "coloring_iterations: {} x 10",
                        run_stats.coloring_iterations
                    ));
                    ui.label(format!(
                        "start_to_pairs_count_time: {:.2}",
                        run_stats.start_to_pairs_count_time.as_secs_f32() * 1000.0
                    ));
                    ui.label(format!(
                        "coloring_fallback_time: {:.2}",
                        run_stats.coloring_fallback_time.as_secs_f32() * 1000.0
                    ));
                });

                ui.collapsing("timestamp queries", |ui| {
                    ui.label(format!(
                        "timestamp_update_mass_props: {:.2}",
                        run_stats.timestamp_update_mass_props
                    ));
                    ui.label(format!(
                        "timestamp_broad_phase: {:.2}",
                        run_stats.timestamp_broad_phase
                    ));
                    ui.label(format!(
                        "timestamp_narrow_phase: {:.2}",
                        run_stats.timestamp_narrow_phase
                    ));
                    ui.label(format!(
                        "timestamp_solver_prep: {:.2}",
                        run_stats.timestamp_solver_prep
                    ));
                    ui.label(format!(
                        "timestamp_solver_solve: {:.2}",
                        run_stats.timestamp_solver_solve
                    ));
                });
            }

            ui.horizontal(|ui| {
                let play_pause_label = if *run_state == RunState::Running {
                    "Pause"
                } else {
                    "Play"
                };
                if ui.button(play_pause_label).clicked() {
                    if *run_state == RunState::Running {
                        *run_state = RunState::Paused;
                    } else {
                        *run_state = RunState::Running;
                    }
                }
                if ui.button("Step").clicked() {
                    *run_state = RunState::Step;
                }
                if ui.button("Restart").clicked() {
                    result.new_selected_demo = Some(*selected_demo);
                }
            });
        });
    });
    result
}
