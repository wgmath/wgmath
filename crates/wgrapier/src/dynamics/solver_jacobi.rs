use crate::dynamics::constraint::GpuTwoBodyConstraint;
use crate::dynamics::prefix_sum::{PrefixSumWorkspace, WgPrefixSum};
use crate::dynamics::{GpuBodySet, GpuSimParams, WgBody, WgSimParams};
use crate::dynamics::{GpuLocalMassProperties, GpuVelocity, GpuWorldMassProperties, WgConstraint};
use wgcore::indirect::DispatchIndirectArgs;
use wgcore::kernel::KernelDispatch;
use wgcore::tensor::{GpuScalar, GpuVector};
use wgcore::Shader;
use wgebra::geometry::WgInv;
use wgparry::math::GpuSim;
use wgparry::queries::GpuIndexedContact;
use wgparry::{dim_shader_defs, substitute_aliases};
use wgpu::{ComputePass, ComputePipeline, Device};

#[derive(Shader)]
#[shader(
    derive(WgSimParams, WgBody, WgConstraint, WgInv),
    src = "solver_jacobi.wgsl",
    src_fn = "substitute_aliases",
    shader_defs = "dim_shader_defs",
    composable = false
)]
pub struct WgSolverJacobi {
    init_constraints: ComputePipeline,
    sort_constraints: ComputePipeline,
    cleanup: ComputePipeline,
    step_jacobi: ComputePipeline,
    step_gauss_seidel: ComputePipeline,
    finalize: ComputePipeline,
    remove_cfm_and_bias: ComputePipeline,
}

pub struct JacobiSolverArgs<'a> {
    pub num_colliders: u32,
    pub contacts: &'a GpuVector<GpuIndexedContact>,
    pub contacts_len: &'a GpuScalar<u32>,
    pub contacts_len_indirect: &'a GpuScalar<DispatchIndirectArgs>,
    pub constraints: &'a GpuVector<GpuTwoBodyConstraint>,
    pub sim_params: &'a GpuScalar<GpuSimParams>,
    pub colliders_len: &'a GpuScalar<u32>,
    pub colliders_len_indirect: &'a GpuScalar<[u32; 3]>,
    pub poses: &'a GpuVector<GpuSim>,
    pub vels: &'a GpuVector<GpuVelocity>,
    pub solver_vels: &'a GpuVector<GpuVelocity>,
    pub solver_vels_out: &'a GpuVector<GpuVelocity>,
    pub mprops: &'a GpuVector<GpuWorldMassProperties>,
    pub body_constraint_counts: &'a GpuVector<u32>,
    pub body_constraint_ids: &'a GpuVector<u32>,
    pub prefix_sum: &'a WgPrefixSum,
    pub prefix_sum_workspace: &'a mut PrefixSumWorkspace,
}

impl WgSolverJacobi {
    const WORKGROUP_SIZE: u32 = 64;

    pub fn dispatch<'a>(
        &self,
        device: &Device,
        pass: &mut ComputePass,
        mut args: JacobiSolverArgs<'a>,
    ) {
        KernelDispatch::new(device, pass, &self.cleanup)
            .bind_at(
                0,
                [
                    (args.solver_vels.buffer(), 3),
                    (args.body_constraint_counts.buffer(), 5),
                ],
            )
            .bind_at(
                1,
                [
                    (args.vels.buffer(), 2),
                    (args.mprops.buffer(), 3),
                    (args.colliders_len.buffer(), 4),
                ],
            )
            .dispatch(args.num_colliders.div_ceil(Self::WORKGROUP_SIZE));

        // Init constraints.
        KernelDispatch::new(device, pass, &self.init_constraints)
            .bind_at(
                0,
                [
                    (args.contacts.buffer(), 0),
                    (args.contacts_len.buffer(), 1),
                    (args.constraints.buffer(), 2),
                    (args.body_constraint_counts.buffer(), 5),
                ],
            )
            .bind(
                1,
                [
                    args.sim_params.buffer(),
                    args.poses.buffer(),
                    args.vels.buffer(),
                    args.mprops.buffer(),
                ],
            )
            .dispatch_indirect(args.contacts_len_indirect.buffer());

        args.prefix_sum.dispatch(
            device,
            pass,
            args.prefix_sum_workspace,
            &args.body_constraint_counts,
        );

        KernelDispatch::new(device, pass, &self.sort_constraints)
            .bind_at(
                0,
                [
                    (args.contacts.buffer(), 0),
                    (args.contacts_len.buffer(), 1),
                    (args.body_constraint_counts.buffer(), 5),
                    (args.body_constraint_ids.buffer(), 6),
                ],
            )
            .dispatch_indirect(args.contacts_len_indirect.buffer());

        // Resolution loop.
        let niters = 10;
        let stabilization_iters = 2;
        let use_jacobi = true;

        for i in 0..niters + stabilization_iters {
            if i == niters {
                KernelDispatch::new(device, pass, &self.remove_cfm_and_bias)
                    .bind_at(
                        0,
                        [
                            (args.contacts_len.buffer(), 1),
                            (args.constraints.buffer(), 2),
                        ],
                    )
                    .dispatch_indirect(args.contacts_len_indirect.buffer());
            }

            if use_jacobi {
                KernelDispatch::new(device, pass, &self.step_jacobi)
                    .bind_at(
                        0,
                        [
                            (args.constraints.buffer(), 2),
                            (args.solver_vels.buffer(), 3),
                            (args.solver_vels_out.buffer(), 4),
                            (args.body_constraint_counts.buffer(), 5),
                            (args.body_constraint_ids.buffer(), 6),
                        ],
                    )
                    .bind_at(1, [(args.colliders_len.buffer(), 4)])
                    .dispatch_indirect(args.colliders_len_indirect.buffer());
                std::mem::swap(&mut args.solver_vels, &mut args.solver_vels_out);
            } else {
                KernelDispatch::new(device, pass, &self.step_gauss_seidel)
                    .bind_at(
                        0,
                        [
                            (args.contacts_len.buffer(), 1),
                            (args.constraints.buffer(), 2),
                            (args.solver_vels.buffer(), 3),
                        ],
                    )
                    .dispatch(1);
            }
        }

        KernelDispatch::new(device, pass, &self.finalize)
            .bind_at(0, [(args.solver_vels.buffer(), 3)])
            .bind_at(
                1,
                [
                    (args.vels.buffer(), 2),
                    (args.mprops.buffer(), 3),
                    (args.colliders_len.buffer(), 4),
                ],
            )
            .dispatch((args.vels.len() as u32).div_ceil(Self::WORKGROUP_SIZE));
    }
}

wgcore::test_shader_compilation!(WgSolverJacobi);
