use crate::bounding_volumes::WgAabb;
use crate::math::GpuSim;
use crate::shapes::{GpuShape, WgShape};
use crate::substitute_aliases;
use na::Vector4;
use wgcore::indirect::{DispatchIndirectArgs, WgIndirect};
use wgcore::kernel::KernelDispatch;
use wgcore::tensor::{GpuScalar, GpuVector};
use wgcore::{test_shader_compilation, Shader};
use wgebra::WgSim3;
use wgpu::{ComputePass, ComputePipeline, Device};

#[derive(Shader)]
#[shader(
    derive(WgSim3, WgShape, WgAabb, WgIndirect),
    src = "./brute_force_broad_phase.wgsl",
    src_fn = "substitute_aliases",
    composable = false
)]
pub struct WgBruteForceBroadPhase {
    main: ComputePipeline,
    reset: ComputePipeline,
    init_indirect_args: ComputePipeline,
    debug_compute_aabb: ComputePipeline, // TODO: remove this. For debugging only.
}

impl WgBruteForceBroadPhase {
    const WORKGROUP_SIZE: u32 = 64;

    /// Dispatch an invocation of [`WgIntegrate::integrate`] for integrating forces and velocities
    /// of every rigid-body in the given [`GpuBodySet`]:
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
        debug_aabb_mins: &GpuVector<Vector4<f32>>,
        debug_aabb_maxs: &GpuVector<Vector4<f32>>,
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
