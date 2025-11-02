//! Brute-force broad-phase collision detection.
//!
//! Tests all pairs of colliders for AABB overlap (O(n²) complexity). While not scalable
//! for large scenes, the algorithm is highly parallelizable on GPU and can outperform more
//! sophisticated algorithms for small to medium-sized simulations (< 1000 objects).
//!
//! The GPU implementation processes pairs in parallel, making effective use of GPU compute
//! resources even though the algorithmic complexity is quadratic.

use crate::bounding_volumes::WgAabb;
use crate::math::GpuSim;
use crate::shapes::{GpuShape, WgShape};
use crate::{dim_shader_defs, substitute_aliases};
use wgcore::indirect::{DispatchIndirectArgs, WgIndirect};

#[cfg(feature = "dim2")]
use nalgebra::Vector2;
#[cfg(feature = "dim3")]
use nalgebra::Vector4;
use wgcore::kernel::KernelDispatch;
use wgcore::tensor::{GpuScalar, GpuVector};
use wgcore::{test_shader_compilation, Shader};
use wgebra::{WgSim2, WgSim3};
use wgpu::{ComputePass, ComputePipeline, Device};

#[derive(Shader)]
#[shader(
    derive(WgSim3, WgSim2, WgShape, WgAabb, WgIndirect),
    src = "./brute_force_broad_phase.wgsl",
    shader_defs = "dim_shader_defs",
    src_fn = "substitute_aliases",
    composable = false
)]
/// GPU shader for brute-force broad-phase collision detection.
///
/// This shader tests all pairs of colliders for AABB overlap in parallel. While O(n²),
/// it can be convenient for testing and debugging more sophisticated algorithms.
pub struct WgBruteForceBroadPhase {
    main: ComputePipeline,
    reset: ComputePipeline,
    init_indirect_args: ComputePipeline,
    debug_compute_aabb: ComputePipeline, // TODO: remove this. For debugging only.
}

impl WgBruteForceBroadPhase {
    const WORKGROUP_SIZE: u32 = 64;

    /// Dispatches the brute-force broad-phase collision detection.
    pub fn dispatch(
        &self,
        device: &Device,
        pass: &mut ComputePass,
        num_colliders: u32,
        poses: &GpuVector<GpuSim>,
        shapes: &GpuVector<GpuShape>,
        num_shapes: &GpuScalar<u32>,
        collision_pairs: &GpuVector<[u32; 2]>,
        collision_pairs_len: &GpuScalar<u32>,
        collision_pairs_indirect: &GpuScalar<DispatchIndirectArgs>,
        #[cfg(feature = "dim2")] debug_aabb_mins: &GpuVector<Vector2<f32>>,
        #[cfg(feature = "dim2")] debug_aabb_maxs: &GpuVector<Vector2<f32>>,
        #[cfg(feature = "dim3")] debug_aabb_mins: &GpuVector<Vector4<f32>>,
        #[cfg(feature = "dim3")] debug_aabb_maxs: &GpuVector<Vector4<f32>>,
    ) {
        KernelDispatch::new(device, pass, &self.reset)
            .bind_at(0, [(collision_pairs_len.buffer(), 4)])
            .dispatch(1);

        KernelDispatch::new(device, pass, &self.main)
            .bind0([
                num_shapes.buffer(),
                poses.buffer(),
                shapes.buffer(),
                collision_pairs.buffer(),
                collision_pairs_len.buffer(),
            ])
            .dispatch(num_colliders.div_ceil(Self::WORKGROUP_SIZE));

        KernelDispatch::new(device, pass, &self.init_indirect_args)
            .bind_at(
                0,
                [
                    (collision_pairs_len.buffer(), 4),
                    (collision_pairs_indirect.buffer(), 5),
                ],
            )
            .dispatch(1);

        KernelDispatch::new(device, pass, &self.debug_compute_aabb)
            .bind_at(0, [(poses.buffer(), 1), (shapes.buffer(), 2)])
            .bind(1, [debug_aabb_mins.buffer(), debug_aabb_maxs.buffer()])
            .dispatch(num_colliders.div_ceil(Self::WORKGROUP_SIZE));
    }
}

test_shader_compilation!(WgBruteForceBroadPhase);
