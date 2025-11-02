//! Warmstarting mechanism for constraint solver temporal coherence.
//!
//! Warmstarting reuses impulses from the previous simulation frame to initialize the
//! constraint solver, significantly improving convergence speed and stability. This exploits
//! temporal coherence - the observation that adjacent frames in a simulation tend to have
//! similar contact configurations and required impulses.
//!
//! # How It Works
//!
//! 1. After solving constraints in frame N, impulse accumulators are stored.
//! 2. In frame N+1, new contacts are matched against old contacts.
//! 3. Matching contacts inherit their previous impulses as starting guesses.
//! 4. The solver converges faster since it starts closer to the solution.

use crate::dynamics::WgConstraint;
use crate::dynamics::{GpuTwoBodyConstraint, GpuTwoBodyConstraintBuilder};
use wgcore::indirect::DispatchIndirectArgs;
use wgcore::kernel::KernelDispatch;
use wgcore::tensor::{GpuScalar, GpuVector};
use wgcore::Shader;
use wgparry::{dim_shader_defs, substitute_aliases};
use wgpu::{ComputePass, ComputePipeline, Device};

/// GPU shader for transferring warmstart impulses between frames.
///
/// This shader matches new contacts against old contacts and transfers impulse
/// accumulators when a match is found.
#[derive(Shader)]
#[shader(
    derive(WgConstraint),
    src = "warmstart.wgsl",
    src_fn = "substitute_aliases",
    shader_defs = "dim_shader_defs",
    composable = false
)]
pub struct WgWarmstart {
    /// Compute pipeline that matches contacts and transfers impulses.
    transfer_warmstart_impulses: ComputePipeline,
}

/// Arguments for warmstart dispatch.
///
/// Contains buffers for both old (previous frame) and new (current frame) constraint data.
#[derive(Copy, Clone)]
pub struct WarmstartArgs<'a> {
    /// Number of contacts in current frame.
    pub contacts_len: &'a GpuScalar<u32>,
    /// Constraint counts per body from previous frame.
    pub old_body_constraint_counts: &'a GpuVector<u32>,
    /// Constraint IDs per body from previous frame.
    pub old_body_constraint_ids: &'a GpuVector<u32>,
    /// Solver constraints from previous frame.
    pub old_constraints: &'a GpuVector<GpuTwoBodyConstraint>,
    /// Constraint builders from previous frame.
    pub old_constraint_builders: &'a GpuVector<GpuTwoBodyConstraintBuilder>,
    /// Solver constraints for current frame (to be warmstarted).
    pub new_constraints: &'a GpuVector<GpuTwoBodyConstraint>,
    /// Constraint builders for current frame.
    pub new_constraint_builders: &'a GpuVector<GpuTwoBodyConstraintBuilder>,
    /// Indirect dispatch arguments based on contact count.
    pub contacts_len_indirect: &'a GpuScalar<DispatchIndirectArgs>,
}

impl WgWarmstart {
    /// Transfers warmstart impulses from old constraints to new constraints.
    ///
    /// This method dispatches a compute shader that searches for matching contacts
    /// between the previous and current frames. When a match is found, the impulse
    /// accumulator from the old contact is copied to the new contact.
    ///
    /// # Parameters
    ///
    /// - `device`: The WebGPU device.
    /// - `pass`: Active compute pass for dispatching.
    /// - `args`: Warmstart arguments containing old and new constraint buffers.
    pub fn transfer_warmstart_impulses<'a>(
        &self,
        device: &Device,
        pass: &mut ComputePass,
        args: WarmstartArgs<'a>,
    ) {
        KernelDispatch::new(device, pass, &self.transfer_warmstart_impulses)
            .bind0([
                args.contacts_len.buffer(),
                args.old_body_constraint_counts.buffer(),
                args.old_body_constraint_ids.buffer(),
                args.old_constraints.buffer(),
                args.old_constraint_builders.buffer(),
                args.new_constraints.buffer(),
                args.new_constraint_builders.buffer(),
            ])
            .dispatch_indirect(args.contacts_len_indirect.buffer());
    }
}
