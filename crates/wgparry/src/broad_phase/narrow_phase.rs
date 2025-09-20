use crate::bounding_volumes::WgAabb;
use crate::math::GpuSim;
use crate::queries::{GpuIndexedContact, WgContact};
use crate::shapes::{GpuShape, WgShape};
use crate::substitute_aliases;
use wgcore::indirect::{DispatchIndirectArgs, WgIndirect};
use wgcore::kernel::KernelDispatch;
use wgcore::tensor::{GpuScalar, GpuVector};
use wgcore::{test_shader_compilation, Shader};
use wgebra::WgSim3;
use wgpu::{ComputePass, ComputePipeline, Device};

#[derive(Shader)]
#[shader(
    derive(WgSim3, WgShape, WgAabb, WgContact, WgIndirect),
    src = "./narrow_phase.wgsl",
    src_fn = "substitute_aliases",
    composable = false
)]
pub struct WgNarrowPhase {
    main: ComputePipeline,
    reset: ComputePipeline,
    init_indirect_args: ComputePipeline,
}

impl WgNarrowPhase {
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
        collision_pairs: &GpuVector<[u32; 2]>,
        collision_pairs_len: &GpuScalar<u32>,
        collision_pairs_indirect: &GpuScalar<DispatchIndirectArgs>,
        contacts: &GpuVector<GpuIndexedContact>,
        contacts_len: &GpuScalar<u32>,
        contacts_indirect: &GpuScalar<DispatchIndirectArgs>,
    ) {
        KernelDispatch::new(device, pass, &self.reset)
            .bind_at(0, [(contacts_len.buffer(), 5)])
            .dispatch(1);

        KernelDispatch::new(device, pass, &self.main)
            .bind0([
                collision_pairs.buffer(),
                collision_pairs_len.buffer(),
                poses.buffer(),
                shapes.buffer(),
                contacts.buffer(),
                contacts_len.buffer(),
            ])
            .dispatch_indirect(collision_pairs_indirect.buffer());

        KernelDispatch::new(device, pass, &self.init_indirect_args)
            .bind_at(
                0,
                [(contacts_len.buffer(), 5), (contacts_indirect.buffer(), 6)],
            )
            .dispatch(1);
    }
}

test_shader_compilation!(WgNarrowPhase);
