//! Narrow-phase collision detection for generating contact manifolds.
//!
//! After the broad-phase identifies potentially colliding pairs using AABBs, the narrow-phase
//! performs detailed collision tests to generate contact manifolds. These manifolds contain
//! precise contact point information needed for physics simulation.
//!
//! The narrow-phase:
//! 1. Takes collision pairs from the broad-phase.
//! 2. Performs shape-specific collision tests (ball-ball, cuboid-cuboid, etc.)
//! 3. Generates contact manifolds with points, normals, and penetration depths.
//! 4. Outputs indexed contacts for the physics solver.

use crate::bounding_volumes::WgAabb;
use crate::math::{GpuSim, Point, Vector};
use crate::queries::{GpuIndexedContact, WgContact};
use crate::shapes::{GpuShape, WgShape};
use crate::{dim_shader_defs, substitute_aliases};
use wgcore::indirect::{DispatchIndirectArgs, WgIndirect};
use wgcore::kernel::KernelDispatch;
use wgcore::tensor::{GpuScalar, GpuVector};
use wgcore::{test_shader_compilation, Shader};
use wgebra::{WgSim2, WgSim3};
use wgpu::{ComputePass, ComputePipeline, Device};

#[derive(Shader)]
#[shader(
    derive(WgSim3, WgSim2, WgShape, WgAabb, WgContact, WgIndirect),
    src = "./narrow_phase.wgsl",
    src_fn = "substitute_aliases",
    shader_defs = "dim_shader_defs",
    composable = false
)]
/// GPU shader for narrow-phase collision detection.
///
/// This shader performs detailed collision tests on potentially colliding pairs identified
/// by the broad-phase. It generates contact manifolds containing:
/// - Contact points (up to 2 in 2D, 4 in 3D)
/// - Contact normals
/// - Penetration depths
///
/// # Pipeline Stages
///
/// The narrow-phase executes in three stages:
/// 1. **Reset**: Clears the contact count from the previous frame.
/// 2. **Main**: Processes collision pairs and generates contacts.
/// 3. **Init indirect args**: Prepares dispatch arguments for subsequent kernels.
pub struct WgNarrowPhase {
    main: ComputePipeline,
    reset: ComputePipeline,
    init_indirect_args: ComputePipeline,
}

impl WgNarrowPhase {
    #[allow(dead_code)]
    const WORKGROUP_SIZE: u32 = 64;

    /// Dispatches the narrow-phase collision detection pipeline.
    ///
    /// # Parameters
    ///
    /// - `device`: The GPU device
    /// - `pass`: The compute pass to record commands into
    /// - `_num_colliders`: Total number of colliders (unused currently)
    /// - `poses`: Collider poses (positions and rotations)
    /// - `shapes`: Collider shapes
    /// - `collision_pairs`: Potentially colliding pairs from broad-phase
    /// - `collision_pairs_len`: Number of collision pairs
    /// - `collision_pairs_indirect`: Indirect dispatch arguments for collision pairs
    /// - `contacts`: Output buffer for contact manifolds
    /// - `contacts_len`: Output count of generated contacts
    /// - `contacts_indirect`: Indirect dispatch arguments for contacts
    pub fn dispatch(
        &self,
        device: &Device,
        pass: &mut ComputePass,
        _num_colliders: u32,
        poses: &GpuVector<GpuSim>,
        shapes: &GpuVector<GpuShape>,
        vertices: &GpuVector<Point<f32>>,
        indices: &GpuVector<u32>,
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
            .bind_at(1, [])
            .bind_at(2, [(vertices.buffer(), 0), (indices.buffer(), 1)])
            .dispatch_indirect(collision_pairs_indirect.buffer());

        KernelDispatch::new(device, pass, &self.init_indirect_args)
            .bind_at(
                0,
                [(contacts_len.buffer(), 5), (contacts_indirect.buffer(), 6)],
            )
            .dispatch(1);
    }
}

test_shader_compilation!(WgNarrowPhase, wgcore, crate::dim_shader_defs());
