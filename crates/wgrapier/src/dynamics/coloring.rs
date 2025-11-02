//! Graph coloring algorithms for parallel constraint solving.
//!
//! This module implements two graph coloring algorithms that enable parallel constraint solving
//! on the GPU:
//!
//! # TOPO-GC (Topological Graph Coloring)
//!
//! A fast, coloring algorithm that typically produces fewer colors and converges
//! in fewer iterations. This is the primary algorithm used by default.
//!
//! **Algorithm**: Iteratively assigns colors to constraints based on local topology. Conflicts
//! are detected and resolved in each iteration until convergence.
//!
//! **Advantages**:
//! - Fast convergence (typically < 10 iterations).
//! - Produces fewer colors (better parallelism for constraints resolution).
//!
//! **Disadvantages**:
//! - May fail to converge for highly complex constraint graphs (if too many colors are needed).
//! - Falls back to Luby if it doesn't converge within iteration limit.
//!
//! # Luby's Algorithm
//!
//! A randomized coloring algorithm used as a fallback when TOPO-GC fails or for very complex
//! constraint graphs.
//!
//! **Algorithm**: Each constraint randomly selects itself or neighbors in each iteration.
//! Selected constraints that don't conflict get the same color.
//!
//! **Advantages**:
//! - Always converges (probabilistically).
//! - Handles arbitrary constraint graphs.
//!
//! **Disadvantages**:
//! - Slower convergence.
//! - May produce more colors (less parallelism for constraints resolution).

use crate::dynamics::{GpuTwoBodyConstraint, WgBody, WgConstraint, WgSimParams};
use crate::pipeline::RunStats;
use wgcore::gpu::GpuInstance;
use wgcore::indirect::DispatchIndirectArgs;
use wgcore::kernel::{CommandEncoderExt, KernelDispatch};
use wgcore::tensor::{GpuScalar, GpuVector};
use wgcore::Shader;
use wgparry::{dim_shader_defs, substitute_aliases};
use wgpu::{BufferAddress, ComputePipeline};

/// GPU shaders for constraint graph coloring.
///
/// Contains compute pipelines for both TOPO-GC and Luby's algorithm.
#[derive(Shader)]
#[shader(
    derive(WgSimParams, WgBody, WgConstraint),
    src = "coloring.wgsl",
    src_fn = "substitute_aliases",
    shader_defs = "dim_shader_defs",
    composable = false
)]
pub struct WgColoring {
    /// Initializes state for Luby's algorithm.
    reset_luby: ComputePipeline,
    /// One iteration of Luby's coloring.
    step_graph_coloring_luby: ComputePipeline,
    /// Initializes state for TOPO-GC algorithm.
    reset_topo_gc: ComputePipeline,
    /// Resets the completion flag for TOPO-GC iteration.
    reset_completion_flag_topo_gc: ComputePipeline,
    /// One iteration of TOPO-GC coloring.
    step_graph_coloring_topo_gc: ComputePipeline,
    /// Detects and fixes conflicts in TOPO-GC coloring.
    fix_conflicts_topo_gc: ComputePipeline,
}

/// Arguments for graph coloring dispatch.
///
/// Contains all GPU buffers needed by the coloring algorithms.
#[derive(Copy, Clone)]
pub struct ColoringArgs<'a> {
    /// Indirect dispatch arguments based on contact count.
    pub contacts_len_indirect: &'a GpuScalar<DispatchIndirectArgs>,
    /// Number of constraints per body.
    pub body_constraint_counts: &'a GpuVector<u32>,
    /// Constraint IDs associated with each body.
    pub body_constraint_ids: &'a GpuVector<u32>,
    /// The constraints to be colored.
    pub constraints: &'a GpuVector<GpuTwoBodyConstraint>,
    /// Output: color assigned to each constraint.
    pub constraints_colors: &'a GpuVector<u32>,
    /// Random values for Luby's algorithm.
    pub constraints_rands: &'a GpuVector<u32>,
    /// Current color being assigned.
    pub curr_color: &'a GpuScalar<u32>,
    /// Count of uncolored constraints (or changed flag for TOPO-GC).
    pub uncolored: &'a GpuScalar<u32>,
    /// Staging buffer for reading uncolored count on CPU.
    pub uncolored_staging: &'a GpuScalar<u32>,
    /// Total number of contacts.
    pub contacts_len: &'a GpuScalar<u32>,
    /// Buffer tracking which constraints are colored.
    pub colored: &'a GpuVector<u32>,
}

impl WgColoring {
    #[allow(dead_code)]
    const WORKGROUP_SIZE: u32 = 64;

    /// Executes Luby's randomized graph coloring algorithm.
    ///
    /// This method runs Luby's algorithm iteratively until all constraints are colored.
    /// Each iteration assigns one color to a subset of constraints.
    ///
    /// # Parameters
    ///
    /// - `gpu`: GPU instance for command submission
    /// - `args`: Coloring arguments containing constraint graph buffers
    /// - `stats`: Statistics structure to record coloring performance
    ///
    /// # Returns
    ///
    /// The total number of colors used. **Note**: Colors are 1-indexed, so valid
    /// color indices are `[1..result]`.
    ///
    /// # CPU-GPU Synchronization
    ///
    /// This method requires CPU-GPU synchronization after each iteration to check
    /// if any constraints remain uncolored, which can add overhead.
    pub async fn dispatch_luby<'a>(
        &self,
        gpu: &GpuInstance,
        args: ColoringArgs<'a>,
        stats: &mut RunStats,
    ) -> u32 {
        let t0 = web_time::Instant::now();
        let queue = gpu.queue();
        let device = gpu.device();
        let mut encoder = device.create_command_encoder(&Default::default());
        let mut pass = encoder.compute_pass("coloring_reset", None);

        KernelDispatch::new(device, &mut pass, &self.reset_luby)
            .bind_at(
                0,
                [
                    (args.constraints_colors.buffer(), 3),
                    (args.constraints_rands.buffer(), 4),
                    (args.contacts_len.buffer(), 7),
                ],
            )
            .dispatch_indirect(args.contacts_len_indirect.buffer());
        drop(pass);
        queue.submit(Some(encoder.finish()));

        let _first_time = 0;
        let mut num_colors = 0;
        for color in 1u32.. {
            let mut encoder = device.create_command_encoder(&Default::default());
            let mut pass = encoder.compute_pass("coloring", None);
            queue.write_buffer(args.curr_color.buffer(), 0, bytemuck::cast_slice(&[color]));
            queue.write_buffer(args.uncolored.buffer(), 0, bytemuck::cast_slice(&[0u32]));
            KernelDispatch::new(device, &mut pass, &self.step_graph_coloring_luby)
                .bind0([
                    args.body_constraint_counts.buffer(),
                    args.body_constraint_ids.buffer(),
                    args.constraints.buffer(),
                    args.constraints_colors.buffer(),
                    args.constraints_rands.buffer(),
                    args.curr_color.buffer(),
                    args.uncolored.buffer(),
                    args.contacts_len.buffer(),
                ])
                .dispatch_indirect(args.contacts_len_indirect.buffer());
            drop(pass);
            encoder.copy_buffer_to_buffer(
                args.uncolored.buffer(),
                0,
                args.uncolored_staging.buffer(),
                0,
                size_of::<u32>() as BufferAddress,
            );
            queue.submit(Some(encoder.finish()));

            let mut uncolored = [0u32];
            args.uncolored_staging
                .read_to(gpu.device(), &mut uncolored)
                .await
                .unwrap();

            if uncolored[0] == 0 {
                num_colors = color + 1;
                break;
            }
        }

        stats.num_colors = num_colors;
        stats.coloring_fallback_time = t0.elapsed();
        num_colors
    }

    /// Executes the TOPO-GC (Topological Graph Coloring) algorithm.
    ///
    /// TOPO-GC is the primary coloring algorithm, typically faster and producing fewer colors
    /// than Luby. It may fail to converge for very complex graphs, in which case it returns
    /// `None` and the caller should fall back to [`dispatch_luby`](Self::dispatch_luby).
    ///
    /// # Parameters
    ///
    /// - `gpu`: GPU instance for command submission
    /// - `args`: Coloring arguments containing constraint graph buffers
    /// - `stats`: Statistics structure to record coloring performance
    ///
    /// # Returns
    ///
    /// - `Some(num_colors)` if coloring succeeded. **Note**: Colors are 1-indexed.
    /// - `None` if the algorithm failed to converge (exceeded iteration limit).
    ///
    /// # Performance Optimization
    ///
    /// To reduce CPU-GPU synchronization overhead, this method batches 10 iterations
    /// per readback. In the future, the batch size could be dynamically adjusted based on previous
    /// frame's convergence rate for better performance.
    pub async fn dispatch_topo_gc<'a>(
        &self,
        gpu: &GpuInstance,
        args: ColoringArgs<'a>,
        stats: &mut RunStats,
    ) -> Option<u32> {
        let t0 = web_time::Instant::now();
        let queue = gpu.queue();
        let device = gpu.device();
        let mut encoder = device.create_command_encoder(&Default::default());
        let mut pass = encoder.compute_pass("coloring_reset", None);

        KernelDispatch::new(device, &mut pass, &self.reset_topo_gc)
            .bind_at(
                0,
                [
                    (args.constraints_colors.buffer(), 3),
                    (args.contacts_len.buffer(), 7),
                ],
            )
            .bind_at(1, [(args.colored.buffer(), 1)])
            .dispatch_indirect(args.contacts_len_indirect.buffer());
        drop(pass);
        queue.submit(Some(encoder.finish()));

        let mut num_loops = 0;
        let mut max_color = [0u32];
        loop {
            num_loops += 1;
            if num_loops > 64 {
                stats.coloring_time = t0.elapsed();
                return None;
            }
            let mut encoder = device.create_command_encoder(&Default::default());
            let mut pass = encoder.compute_pass("coloring", None);

            // PERF: we queue multiple passes directly to reduce the frequency when we have to read
            //       the stop flag on the CPU side (it has a huge overhead, not entirely sure why).
            // TODO PERF: we could auto-adjust the number of inner loops based on the last frame’s
            //            color count. That way we can reduce to a minimum the number of times we
            //            end up having to run more than a single inner loop before convergence.
            for _ in 0..10 {
                KernelDispatch::new(device, &mut pass, &self.reset_completion_flag_topo_gc)
                    .bind0([])
                    .bind_at(1, [(args.uncolored.buffer(), 0)])
                    .dispatch(1);
                KernelDispatch::new(device, &mut pass, &self.step_graph_coloring_topo_gc)
                    .bind_at(
                        0,
                        [
                            (args.body_constraint_counts.buffer(), 0),
                            (args.body_constraint_ids.buffer(), 1),
                            (args.constraints.buffer(), 2),
                            (args.constraints_colors.buffer(), 3),
                            (args.contacts_len.buffer(), 7),
                        ],
                    )
                    .bind_at(
                        1,
                        [(args.uncolored.buffer(), 0), (args.colored.buffer(), 1)],
                    )
                    .dispatch_indirect(args.contacts_len_indirect.buffer());
                KernelDispatch::new(device, &mut pass, &self.fix_conflicts_topo_gc)
                    .bind_at(
                        0,
                        [
                            (args.body_constraint_counts.buffer(), 0),
                            (args.body_constraint_ids.buffer(), 1),
                            (args.constraints.buffer(), 2),
                            (args.constraints_colors.buffer(), 3),
                            (args.contacts_len.buffer(), 7),
                        ],
                    )
                    .bind_at(
                        1,
                        [(args.uncolored.buffer(), 0), (args.colored.buffer(), 1)],
                    )
                    .dispatch_indirect(args.contacts_len_indirect.buffer());
            }
            drop(pass);
            encoder.copy_buffer_to_buffer(
                args.uncolored.buffer(),
                0,
                args.uncolored_staging.buffer(),
                0,
                size_of::<u32>() as BufferAddress,
            );
            queue.submit(Some(encoder.finish()));

            args.uncolored_staging
                .read_to(gpu.device(), &mut max_color)
                .await
                .unwrap();

            if max_color[0] != 0 {
                break;
            }
        }

        stats.coloring_time = t0.elapsed();
        stats.num_colors = max_color[0]; // Don’t add 1 for the color count.
        stats.coloring_iterations = num_loops;

        Some(max_color[0] + 1) // NOTE: color indices are 1-based.
    }
}
