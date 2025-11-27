//! Physics simulation pipeline orchestrating broad-phase, narrow-phase, and constraint solving.
//!
//! This module provides the high-level physics pipeline that coordinates all stages of a physics
//! simulation step on the GPU. The pipeline manages collision detection, contact generation,
//! constraint solving, and integration.

use crate::dynamics::body::{GpuLocalMassProperties, GpuVelocity, GpuWorldMassProperties};
use crate::dynamics::{
    prefix_sum::{PrefixSumWorkspace, WgPrefixSum},
    ColoringArgs, GpuImpulseJointSet, GpuSimParams, GpuTwoBodyConstraint,
    GpuTwoBodyConstraintBuilder, JointSolverArgs, SolverArgs, WarmstartArgs, WgColoring,
    WgJointSolver, WgMpropsUpdate, WgSolver, WgWarmstart,
};
use crate::wgparry::{
    broad_phase::{Lbvh, WgBruteForceBroadPhase, WgNarrowPhase},
    queries::GpuIndexedContact,
    shapes::GpuShape,
};
use naga_oil::compose::ComposerError;
use nalgebra::Vector4;
use rapier::dynamics::{ImpulseJointSet, RigidBodySet};
use rapier::geometry::ColliderSet;
use std::collections::HashMap;
use std::time::Duration;
use wgcore::gpu::GpuInstance;
use wgcore::indirect::DispatchIndirectArgs;
use wgcore::kernel::{CommandEncoderExt, KernelDispatch};
use wgcore::tensor::{GpuScalar, GpuVector};
use wgcore::timestamps::GpuTimestamps;
use wgcore::Shader;
use wgparry::broad_phase::LbvhState;
use wgparry::math::{GpuSim, Point};
use wgparry::shapes::ShapeBuffers;
use wgpu::{BufferUsages, Device};

/// Performance statistics collected during a physics simulation step.
///
/// This structure tracks timing and iteration counts for various stages of the physics pipeline,
/// useful for profiling and optimization.
#[derive(Default, Copy, Clone, Debug)]
pub struct RunStats {
    /// Number of colors used in the graph coloring algorithm for parallel constraint solving.
    pub num_colors: u32,
    /// Duration from the start of the step until collision pair count is read back from GPU.
    pub start_to_pairs_count_time: Duration,
    /// Time spent on the graph coloring algorithm.
    pub coloring_time: Duration,
    /// Number of iterations the coloring algorithm took to converge.
    pub coloring_iterations: u32,
    /// Time spent on the fallback coloring method (if the primary method failed).
    pub coloring_fallback_time: Duration,
    /// Total simulation time including GPU-to-CPU readbacks.
    pub total_simulation_time_with_readback: Duration,
    /// GPU timestamp for updating the mass properties.
    pub timestamp_update_mass_props: f64,
    /// GPU timestamp for the broad-phase collision detection.
    pub timestamp_broad_phase: f64,
    /// GPU timestamp for the narrow-phase contact generation.
    pub timestamp_narrow_phase: f64,
    /// GPU timestamp for constraint solver preparation.
    pub timestamp_solver_prep: f64,
    /// GPU timestamp for the constraint solver.
    pub timestamp_solver_solve: f64,
}

impl RunStats {
    /// Returns the total simulation time in milliseconds.
    pub fn total_simulation_time_ms(&self) -> f32 {
        self.total_simulation_time_with_readback.as_secs_f32() * 1000.0
    }
}

/// GPU-resident physics simulation state containing all rigid bodies, shapes, and solver data.
///
/// This structure holds all the buffers needed for a complete physics simulation on the GPU:
/// - Rigid body poses, velocities, and mass properties
/// - Collision shapes and contact data
/// - Constraints and solver state
/// - Auxiliary data structures (LBVH, prefix sum workspace, etc.)
///
/// The state can be initialized from CPU-side Rapier data structures and then updated
/// entirely on the GPU each frame.
pub struct GpuPhysicsState {
    sim_params: GpuScalar<GpuSimParams>,
    poses: GpuVector<GpuSim>,
    local_mprops: GpuVector<GpuLocalMassProperties>,
    mprops: GpuVector<GpuWorldMassProperties>,
    vels: GpuVector<GpuVelocity>,
    solver_vels: GpuVector<GpuVelocity>,
    solver_vels_out: GpuVector<GpuVelocity>,
    solver_vels_inc: GpuVector<GpuVelocity>,
    vertex_buffers: GpuVector<Point<f32>>,
    index_buffers: GpuVector<u32>,
    shapes: GpuVector<GpuShape>,
    num_shapes: GpuScalar<u32>,
    num_shapes_indirect: GpuScalar<[u32; 3]>,
    collision_pairs: GpuVector<[u32; 2]>,
    collision_pairs_len: GpuScalar<u32>,
    collision_pairs_len_staging: GpuScalar<u32>,
    collision_pairs_indirect: GpuScalar<DispatchIndirectArgs>,
    contacts: GpuVector<GpuIndexedContact>,
    contacts_len: GpuScalar<u32>,
    contacts_indirect: GpuScalar<DispatchIndirectArgs>,
    new_constraints: GpuVector<GpuTwoBodyConstraint>,
    new_constraint_builders: GpuVector<GpuTwoBodyConstraintBuilder>,
    new_constraints_counts: GpuVector<u32>,
    new_body_constraint_ids: GpuVector<u32>,
    old_constraints: GpuVector<GpuTwoBodyConstraint>,
    old_constraint_builders: GpuVector<GpuTwoBodyConstraintBuilder>,
    old_constraints_counts: GpuVector<u32>,
    old_body_constraint_ids: GpuVector<u32>,
    constraints_colors: GpuVector<u32>,
    colored: GpuVector<u32>,
    constraints_rands: GpuVector<u32>,
    curr_color: GpuScalar<u32>,
    uncolored: GpuScalar<u32>,
    uncolored_staging: GpuScalar<u32>,
    lbvh: LbvhState,
    joints: GpuImpulseJointSet,

    prefix_sum_workspace: PrefixSumWorkspace,

    #[allow(dead_code)]
    debug_aabb_mins: GpuVector<Vector4<f32>>,
    #[allow(dead_code)]
    debug_aabb_maxs: GpuVector<Vector4<f32>>,
}

impl GpuPhysicsState {
    /// Creates a new GPU physics state from CPU-side Rapier data structures.
    ///
    /// This method extracts rigid body and collider data from Rapier's CPU representations
    /// and uploads them to GPU buffers. Each collider is treated as a separate rigid body
    /// in the GPU simulation.
    ///
    /// # Parameters
    ///
    /// - `device`: The WebGPU device used to allocate GPU buffers.
    /// - `bodies`: The set of rigid bodies from Rapier.
    /// - `colliders`: The set of colliders from Rapier.
    ///
    /// # Panics
    ///
    /// Panics if any rigid body has more than one collider attached, as this is not currently supported.
    ///
    /// # GPU Memory Allocation
    ///
    /// This method allocates significant GPU memory for:
    /// - Body poses, velocities, and mass properties.
    /// - Collision shapes and contact buffers.
    /// - Constraint solver data structures.
    /// - LBVH acceleration structure.
    pub fn from_rapier(
        device: &Device,
        bodies: &RigidBodySet,
        colliders: &ColliderSet,
        impulse_joints: &ImpulseJointSet,
        use_jacobi: bool,
    ) -> Self {
        let mut rb_poses = Vec::new();
        let mut rb_local_mprops = Vec::new();
        let mut rb_mprops = Vec::new();
        let mut shapes = Vec::new();
        let mut shape_buffers = ShapeBuffers::default();
        let mut body_ids = HashMap::new();

        for (_, co) in colliders.iter() {
            let parent = co.parent().map(|h| &bodies[h]);

            if let Some(parent) = parent {
                assert_eq!(
                    parent.colliders().len(),
                    1,
                    "Only bodies with exactly one collider are supported."
                );
            }

            let mut local_mprops = GpuLocalMassProperties::default();
            let mut mprops = GpuWorldMassProperties {
                com: parent.map(|body| *body.translation()).unwrap_or_default(), // TODO: is this still needed?
                ..Default::default()
            };
            if parent.map(|b| !b.is_dynamic()).unwrap_or(true) {
                local_mprops.inv_mass.fill(0.0);
                local_mprops.inv_principal_inertia = nalgebra::zero();
                mprops.inv_mass.fill(0.0);
                mprops.inv_inertia = nalgebra::zero();
            }

            if let Some(h) = co.parent() {
                let id = rb_poses.len();
                body_ids.insert(h, id as u32);
            }

            rb_local_mprops.push(local_mprops);
            rb_mprops.push(mprops);
            shapes.push(
                GpuShape::from_parry(co.shape(), &mut shape_buffers).expect("Unsupported shape"),
            );
            #[cfg(feature = "dim2")]
            rb_poses.push(GpuSim::from(*co.position()));
            #[cfg(feature = "dim3")]
            rb_poses.push(GpuSim::from_isometry(*co.position(), 1.0));
        }

        // NOTE: wgpu doesn’t like empty storage buffer bindings.
        //       So if the vertex/index buffers are empty, add some dummy value instead of leaving
        //       them empty. This won’t have any performance impact.
        if shape_buffers.vertices.is_empty() {
            shape_buffers.vertices.push(Point::origin());
        }
        if shape_buffers.indices.is_empty() {
            shape_buffers.indices.extend_from_slice(&[0; 3]);
        }

        let vertex_buffers =
            GpuVector::encase(device, &shape_buffers.vertices, BufferUsages::STORAGE);
        let index_buffers = GpuVector::init(device, &shape_buffers.indices, BufferUsages::STORAGE);

        let joints = GpuImpulseJointSet::from_rapier(device, impulse_joints, &body_ids);

        let num_bodies = rb_poses.len();
        let rb_vels = vec![GpuVelocity::default(); num_bodies];
        let storage: BufferUsages = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
        let shapes = GpuVector::init(device, &shapes, storage);
        let num_shapes = GpuScalar::init(device, num_bodies as u32, BufferUsages::UNIFORM);
        let num_shapes_indirect = GpuScalar::init(
            device,
            [num_bodies.div_ceil(64) as u32, 1, 1],
            BufferUsages::STORAGE | BufferUsages::INDIRECT,
        );

        const DEFAULT_CONTACT_COUNTS: u32 = 1024; // NOTE: this will be resized automatically.
        let collision_pairs = GpuVector::uninit(device, DEFAULT_CONTACT_COUNTS, storage);
        let collision_pairs_len =
            GpuScalar::uninit(device, BufferUsages::STORAGE | BufferUsages::COPY_SRC);
        let collision_pairs_len_staging =
            GpuScalar::uninit(device, BufferUsages::MAP_READ | BufferUsages::COPY_DST);
        let collision_pairs_indirect =
            GpuScalar::uninit(device, BufferUsages::STORAGE | BufferUsages::INDIRECT);

        let contacts = GpuVector::uninit_encased(device, DEFAULT_CONTACT_COUNTS, storage);
        let contacts_len = GpuScalar::uninit(device, storage);
        let contacts_indirect =
            GpuScalar::uninit(device, BufferUsages::STORAGE | BufferUsages::INDIRECT);
        let old_constraints = GpuVector::uninit_encased(device, DEFAULT_CONTACT_COUNTS, storage);
        let old_constraint_builders =
            GpuVector::uninit_encased(device, DEFAULT_CONTACT_COUNTS, storage);
        let new_constraints = GpuVector::uninit_encased(device, DEFAULT_CONTACT_COUNTS, storage);
        let new_constraint_builders =
            GpuVector::uninit_encased(device, DEFAULT_CONTACT_COUNTS, storage);
        let constraints_colors = GpuVector::uninit_encased(device, DEFAULT_CONTACT_COUNTS, storage);
        let colored = GpuVector::uninit_encased(device, DEFAULT_CONTACT_COUNTS, storage);
        let constraints_rands = GpuVector::uninit_encased(device, DEFAULT_CONTACT_COUNTS, storage);
        let old_constraints_counts = GpuVector::uninit_encased(device, num_bodies as u32, storage);
        let new_constraints_counts = GpuVector::uninit_encased(device, num_bodies as u32, storage);
        let old_body_constraint_ids =
            GpuVector::uninit_encased(device, DEFAULT_CONTACT_COUNTS * 2, storage);
        let new_body_constraint_ids =
            GpuVector::uninit_encased(device, DEFAULT_CONTACT_COUNTS * 2, storage);
        let mut sim_params = if use_jacobi {
            GpuSimParams::jacobi()
        } else {
            GpuSimParams::tgs_soft()
        };
        sim_params.dt /= sim_params.num_solver_iterations as f32;

        Self {
            sim_params: GpuScalar::init(
                device,
                sim_params,
                BufferUsages::STORAGE | BufferUsages::UNIFORM,
            ),
            vels: GpuVector::encase(device, &rb_vels, storage),
            solver_vels: GpuVector::encase(device, &rb_vels, storage),
            solver_vels_out: GpuVector::encase(device, &rb_vels, storage),
            solver_vels_inc: GpuVector::encase(device, &rb_vels, storage),
            joints,
            local_mprops: GpuVector::encase(device, &rb_local_mprops, storage),
            mprops: GpuVector::encase(device, &rb_mprops, storage),
            poses: GpuVector::init(
                device,
                &rb_poses,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ),
            vertex_buffers,
            index_buffers,
            shapes,
            num_shapes,
            num_shapes_indirect,
            collision_pairs,
            collision_pairs_len,
            collision_pairs_len_staging,
            collision_pairs_indirect,
            contacts,
            contacts_len,
            contacts_indirect,
            old_constraints,
            old_constraint_builders,
            old_constraints_counts,
            new_constraints,
            new_constraint_builders,
            new_constraints_counts,
            constraints_colors,
            colored,
            constraints_rands,
            curr_color: GpuScalar::init(
                device,
                0,
                BufferUsages::STORAGE
                    | BufferUsages::UNIFORM
                    | BufferUsages::COPY_DST
                    | BufferUsages::COPY_SRC,
            ),
            uncolored: GpuScalar::init(
                device,
                0,
                BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            ),
            uncolored_staging: GpuScalar::init(
                device,
                0,
                BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            ),
            old_body_constraint_ids,
            new_body_constraint_ids,
            prefix_sum_workspace: PrefixSumWorkspace::default(),
            debug_aabb_mins: GpuVector::uninit(device, num_bodies as u32, storage),
            debug_aabb_maxs: GpuVector::uninit(device, num_bodies as u32, storage),
            lbvh: LbvhState::new(device).unwrap(),
        }
    }

    /// Returns a reference to the GPU buffer containing rigid body poses.
    ///
    /// The poses are represented as similarity transformations (position + rotation + scale)
    /// in world space.
    pub fn poses(&self) -> &GpuVector<GpuSim> {
        &self.poses
    }

    /// The set of joints part of the simulation.
    pub fn joints(&self) -> &GpuImpulseJointSet {
        &self.joints
    }

    /// Returns a reference to the GPU buffer containing collision shapes.
    ///
    /// Each shape corresponds to one rigid body in the simulation.
    pub fn shapes(&self) -> &GpuVector<GpuShape> {
        &self.shapes
    }
}

/// The main GPU physics pipeline coordinating all simulation stages.
///
/// This structure contains all the compute shaders needed to run a complete physics simulation
/// on the GPU. It orchestrates the following stages in each simulation step:
///
/// 1. **Gravity application**: Updates velocities with gravitational forces.
/// 2. **Broad-phase**: Uses LBVH to find potentially colliding pairs.
/// 3. **Narrow-phase**: Generates detailed contact information for collision pairs.
/// 4. **Constraint preparation**: Converts contacts into solver constraints.
/// 5. **Graph coloring**: Colors constraints to enable parallel solving.
/// 6. **Constraint solving**: Iteratively solves constraints using TGS or PGS.
/// 7. **Integration**: Updates poses based on solved velocities.
pub struct GpuPhysicsPipeline {
    gravity: WgMpropsUpdate,
    #[allow(dead_code)]
    broad_phase: WgBruteForceBroadPhase,
    narrow_phase: WgNarrowPhase,
    solver: WgSolver,
    joint_solver: WgJointSolver,
    prefix_sum: WgPrefixSum,
    lbvh: Lbvh,
    coloring: WgColoring,
    warmstart: WgWarmstart,
}

impl GpuPhysicsPipeline {
    /// Creates a new physics pipeline from a WebGPU device.
    ///
    /// This method compiles all the compute shaders needed for the physics simulation and
    /// creates the compute pipelines.
    ///
    /// # Parameters
    ///
    /// - `device`: The WebGPU device used for shader compilation
    ///
    /// # Returns
    ///
    /// Returns `Ok(Self)` if all shaders compiled successfully, or a [`ComposerError`]
    /// if shader compilation failed.
    pub fn from_device(device: &Device) -> Result<Self, ComposerError> {
        Ok(Self {
            gravity: WgMpropsUpdate::from_device(device)?,
            broad_phase: WgBruteForceBroadPhase::from_device(device)?,
            narrow_phase: WgNarrowPhase::from_device(device)?,
            solver: WgSolver::from_device(device)?,
            joint_solver: WgJointSolver::from_device(device)?,
            prefix_sum: WgPrefixSum::from_device(device)?,
            lbvh: Lbvh::from_device(device)?,
            coloring: WgColoring::from_device(device)?,
            warmstart: WgWarmstart::from_device(device)?,
        })
    }

    /// Executes one physics simulation timestep on the GPU.
    ///
    /// This method runs the complete physics pipeline:
    /// 1. Update world-space mass-properties.
    /// 2. Builds LBVH and finds collision pairs (broad-phase).
    /// 3. Generates contact manifolds (narrow-phase).
    /// 4. Prepares solver constraints from contacts.
    /// 5. Colors constraints for parallel solving.
    /// 6. Solves constraints iteratively using TGS.
    /// 7. Integrates velocities to update poses.
    ///
    /// # Buffer Resizing
    ///
    /// If the number of collision pairs exceeds buffer capacity, this method automatically
    /// allocates larger buffers (next power of two) and re-runs the broad-phase.
    pub async fn step(
        &self,
        gpu: &GpuInstance,
        state: &mut GpuPhysicsState,
        mut timestamps: Option<&mut GpuTimestamps>,
        use_jacobi: bool,
    ) -> RunStats {
        let mut stats = RunStats::default();
        let t_phase1 = web_time::Instant::now();
        let mut encoder = gpu.device().create_command_encoder(&Default::default());
        let mut pass = encoder.compute_pass("step_simulation", timestamps.as_deref_mut());
        KernelDispatch::new(gpu.device(), &mut pass, &self.gravity.main)
            .bind0([
                state.mprops.buffer(),
                state.local_mprops.buffer(),
                state.poses.buffer(),
            ])
            .dispatch(state.poses.len().div_ceil(64) as u32);
        drop(pass);

        let mut pass = encoder.compute_pass("lbvh", timestamps.as_deref_mut());

        // state.broad_phase.dispatch(
        //     gpu.device(),
        //     &mut pass,
        //     state.poses.len() as u32,
        //     &state.poses,
        //     &state.shapes,
        //     &state.num_shapes,
        //     &state.collision_pairs,
        //     &state.collision_pairs_len,
        //     &state.collision_pairs_indirect,
        //     &state.debug_aabb_mins,
        //     &state.debug_aabb_maxs,
        // );

        self.lbvh.update_tree(
            gpu.device(),
            &mut pass,
            &mut state.lbvh,
            state.poses.len() as u32,
            &state.poses,
            &state.vertex_buffers,
            &state.shapes,
            &state.num_shapes,
        );
        self.lbvh.find_pairs(
            gpu.device(),
            &mut pass,
            &mut state.lbvh,
            state.poses.len() as u32,
            &state.num_shapes,
            &state.collision_pairs,
            &state.collision_pairs_len,
            &state.collision_pairs_indirect,
        );
        drop(pass);

        state
            .collision_pairs_len_staging
            .copy_from(&mut encoder, &state.collision_pairs_len);

        gpu.queue().submit(Some(encoder.finish()));

        let mut num_collision_pairs = [0u32];
        state
            .collision_pairs_len_staging
            .read_to(gpu.device(), &mut num_collision_pairs)
            .await
            .unwrap();
        let num_collision_pairs = num_collision_pairs[0];
        stats.start_to_pairs_count_time = t_phase1.elapsed();
        let mut encoder = gpu.device().create_command_encoder(&Default::default());

        // TODO PERF: since we are reading the num_collision_pairs anyway for the sake of buffer resizing,
        //            we might as well just use this for dispatch instead of doing indirect dispatch
        //            (and thus remove `collision_pairs_indirect`).
        if num_collision_pairs >= state.collision_pairs.len() as u32 {
            let storage: BufferUsages = BufferUsages::STORAGE | BufferUsages::COPY_SRC;

            // The collision buffers are too small, resize them.
            let desired_len = num_collision_pairs.next_power_of_two();

            // println!(
            //     "REALLOCATING BUFFERS. Need {}, found {}, allocating: {}",
            //     num_collision_pairs,
            //     state.collision_pairs.len(),
            //     desired_len,
            // );

            // TODO: encapsulate that somewhere
            state.collision_pairs = GpuVector::uninit(gpu.device(), desired_len, storage);
            state.contacts = GpuVector::uninit_encased(gpu.device(), desired_len, storage);
            state.old_constraints = GpuVector::uninit_encased(gpu.device(), desired_len, storage);
            state.old_constraint_builders =
                GpuVector::uninit_encased(gpu.device(), desired_len, storage);
            state.old_body_constraint_ids =
                GpuVector::uninit_encased(gpu.device(), desired_len * 2, storage);
            state.new_constraints = GpuVector::uninit_encased(gpu.device(), desired_len, storage);
            state.new_constraint_builders =
                GpuVector::uninit_encased(gpu.device(), desired_len, storage);
            state.new_body_constraint_ids =
                GpuVector::uninit_encased(gpu.device(), desired_len * 2, storage);
            state.constraints_colors =
                GpuVector::uninit_encased(gpu.device(), desired_len, storage);
            state.colored = GpuVector::uninit_encased(gpu.device(), desired_len, storage);
            state.constraints_rands = GpuVector::uninit_encased(gpu.device(), desired_len, storage);

            // Re-run the broad-phase with the correct buffer lengths.
            let mut pass = encoder.compute_pass("lbvh-after-resize", None);
            self.lbvh.find_pairs(
                gpu.device(),
                &mut pass,
                &mut state.lbvh,
                state.poses.len() as u32,
                &state.num_shapes,
                &state.collision_pairs,
                &state.collision_pairs_len,
                &state.collision_pairs_indirect,
            );
            drop(pass);
        }

        let mut pass = encoder.compute_pass("narrow phase", timestamps.as_deref_mut());
        self.narrow_phase.dispatch(
            gpu.device(),
            &mut pass,
            state.poses.len() as u32,
            &state.poses,
            &state.shapes,
            &state.vertex_buffers,
            &state.index_buffers,
            &state.collision_pairs,
            &state.collision_pairs_len,
            &state.collision_pairs_indirect,
            &state.contacts,
            &state.contacts_len,
            &state.contacts_indirect,
        );
        drop(pass);

        let mut pass = encoder.compute_pass("jacobi prep", timestamps.as_deref_mut());
        let mut solver_args = SolverArgs {
            num_colliders: state.poses.len() as u32,
            contacts: &state.contacts,
            contacts_len: &state.contacts_len,
            contacts_len_indirect: &state.contacts_indirect,
            constraints: &state.new_constraints,
            constraint_builders: &state.new_constraint_builders,
            sim_params: &state.sim_params,
            colliders_len: &state.num_shapes,
            colliders_len_indirect: &state.num_shapes_indirect,
            poses: &state.poses,
            vels: &state.vels,
            solver_vels: &state.solver_vels,
            solver_vels_out: &state.solver_vels_out,
            solver_vels_inc: &state.solver_vels_inc,
            mprops: &state.mprops,
            local_mprops: &state.local_mprops,
            body_constraint_counts: &state.new_constraints_counts,
            body_constraint_ids: &state.new_body_constraint_ids,
            constraints_colors: &state.constraints_colors,
            curr_color: &state.curr_color,
            prefix_sum: &self.prefix_sum,
            num_colors: 0,
        };
        let joint_solver_args = JointSolverArgs {
            sim_params: &state.sim_params,
            poses: &state.poses,
            mprops: &state.mprops,
            local_mprops: &state.local_mprops,
            joints: &state.joints,
            solver_vels: &state.solver_vels,
        };

        self.solver.prepare(
            gpu.device(),
            &mut pass,
            solver_args,
            &mut state.prefix_sum_workspace,
        );

        // NOTE: if webgpu allowed to (but it doesn’t), we could run this kernel completely in parallel of the graph coloring.
        let warmstart_args = WarmstartArgs {
            contacts_len: &state.contacts_len,
            old_body_constraint_counts: &state.old_constraints_counts,
            old_constraint_builders: &state.old_constraint_builders,
            old_body_constraint_ids: &state.old_body_constraint_ids,
            old_constraints: &state.old_constraints,
            new_constraints: &state.new_constraints,
            new_constraint_builders: &state.new_constraint_builders,
            contacts_len_indirect: &state.contacts_indirect,
        };

        if !use_jacobi {
            self.warmstart
                .transfer_warmstart_impulses(gpu.device(), &mut pass, warmstart_args);
        }

        drop(pass);

        gpu.queue().submit(Some(encoder.finish()));

        // self.slow_verify_collision_pair_lists(gpu, state).await;

        // NOTE: jacobi doesn’t need graph coloring.
        if !use_jacobi {
            let coloring_args = ColoringArgs {
                contacts_len_indirect: &state.contacts_indirect,
                body_constraint_counts: &state.new_constraints_counts,
                body_constraint_ids: &state.new_body_constraint_ids,
                constraints: &state.new_constraints,
                constraints_colors: &state.constraints_colors,
                constraints_rands: &state.constraints_rands,
                curr_color: &state.curr_color,
                uncolored: &state.uncolored,
                uncolored_staging: &state.uncolored_staging,
                contacts_len: &state.contacts_len,
                colored: &state.colored,
            };

            if let Some(num_colors) = self
                .coloring
                .dispatch_topo_gc(gpu, coloring_args, &mut stats)
                .await
            {
                solver_args.num_colors = num_colors;
            } else {
                solver_args.num_colors = self
                    .coloring
                    .dispatch_luby(gpu, coloring_args, &mut stats)
                    .await;
            }

            stats.num_colors = solver_args.num_colors;
        }

        // // gpu.queue().submit(Some(encoder.finish()));
        // println!("Found collision pairs: {}", num_collision_pairs);
        // return;

        let mut encoder = gpu.device().create_command_encoder(&Default::default());
        let mut pass = encoder.compute_pass("solve", timestamps);
        self.solver.solve_tgs(
            gpu.device(),
            &mut pass,
            &self.joint_solver,
            solver_args,
            joint_solver_args,
            use_jacobi,
        );
        drop(pass);
        gpu.queue().submit(Some(encoder.finish()));
        // println!("Simulation time: {}.", t0.elapsed().as_secs_f32() * 1000.0);

        // Swap buffers.
        std::mem::swap(&mut state.old_constraints, &mut state.new_constraints);
        std::mem::swap(
            &mut state.old_constraint_builders,
            &mut state.new_constraint_builders,
        );
        std::mem::swap(
            &mut state.old_body_constraint_ids,
            &mut state.new_body_constraint_ids,
        );
        std::mem::swap(
            &mut state.old_constraints_counts,
            &mut state.new_constraints_counts,
        );

        stats
    }

    /// Debugging helper to verify collision pair lists on the CPU.
    ///
    /// This method reads back all collision data from the GPU and validates the constraint
    /// graph structure, checking that constraints are properly associated with bodies.
    /// It's useful for debugging broad-phase or constraint solver issues.
    ///
    /// # Performance Warning
    ///
    /// This method performs extensive CPU-GPU synchronization and should only be used
    /// for debugging, never in production code.
    #[allow(dead_code)] // Very helpful piece of code for debugging the broad-phase's result.
    async fn slow_verify_collision_pair_lists(
        &self,
        gpu: &GpuInstance,
        state: &mut GpuPhysicsState,
    ) {
        let new_poses = state.poses.slow_read(gpu).await;
        let num_constraints = state.contacts_len.slow_read(gpu).await[0];
        let num_collision_pairs = state.collision_pairs_len.slow_read(gpu).await[0];
        let ids = state.new_body_constraint_ids.slow_read(gpu).await;
        let mut counts = state.new_constraints_counts.slow_read(gpu).await;
        println!("First constraint id: {}", ids[0]);
        println!("Num constraints: {}", num_constraints);
        println!("Num collision pairs: {}", num_collision_pairs);

        {
            counts.insert(0, 0);
            let graph: HashMap<usize, Vec<u32>> = counts
                .windows(2)
                .enumerate()
                .map(|(body_id, rng)| {
                    let idx = ids[rng[0] as usize..rng[1] as usize].to_vec();
                    (body_id, idx)
                })
                .collect();
            let mut c2b = vec![vec![]; num_constraints as usize];

            println!("Num constraints: {}", num_constraints);
            for (bid, constraints) in graph.iter() {
                for cid in constraints {
                    c2b[*cid as usize].push(*bid);
                    assert!(
                        c2b[*cid as usize].len() <= 2,
                        "Constraint {} involves: {:?}",
                        *cid,
                        &c2b[*cid as usize]
                    );
                }
            }

            // let mut colors = vec![-1; num_constraints as usize];
            //
            // for cid in 0..num_constraints {}

            for (k, w) in counts.windows(2).enumerate() {
                if w[1] - w[0] > 64 {
                    println!(
                        "Significant count for body {}: [{},{}] = {}. pose: {:?}",
                        k,
                        w[0],
                        w[1],
                        w[1] - w[0],
                        new_poses[k]
                    );
                }
            }
        }

        for pos in &new_poses {
            #[allow(clippy::eq_op)] // We want to check for nans.
            if pos != pos {
                println!("###### Incorrect new pos: {:?}", pos);
            }
        }
    }
}
