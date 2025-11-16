use crate::dynamics::{
    GpuLocalMassProperties, GpuSimParams, GpuVelocity, GpuWorldMassProperties, WgBody, WgSimParams,
};
use encase::ShaderType;
use nalgebra::Vector2;
use rapier::dynamics::{
    GenericJoint, ImpulseJoint, ImpulseJointSet, JointLimits, JointMotor, RigidBodyHandle,
};
use rapier::math::SPATIAL_DIM;
use rapier::prelude::MotorModel;
use std::collections::HashMap;
use wgcore::kernel::KernelDispatch;
use wgcore::tensor::{GpuScalar, GpuVector};
use wgcore::Shader;
use wgebra::{WgQuat, WgRot2, WgSim2, WgSim3};
use wgparry::math::{AngVector, GpuSim, Vector};
use wgparry::{dim_shader_defs, substitute_aliases};
use wgpu::{BufferUsages, ComputePass, ComputePipeline, Device};

#[cfg(feature = "dim2")]
use wgebra::GpuSim2;
#[cfg(feature = "dim3")]
use {
    nalgebra::{Similarity3, Vector4},
    wgebra::GpuSim3,
};

#[derive(Copy, Clone, Debug, ShaderType)]
pub(crate) struct GpuImpulseJoint {
    body_a: u32,
    body_b: u32,
    data: GpuGenericJoint,
}

impl GpuImpulseJoint {
    pub fn from_rapier(joint: &ImpulseJoint, body_id: &HashMap<RigidBodyHandle, u32>) -> Self {
        Self {
            body_a: body_id[&joint.body1],
            body_b: body_id[&joint.body2],
            data: joint.data.into(),
        }
    }
}

#[derive(Copy, Clone, Debug, ShaderType)]
#[cfg(feature = "dim2")]
struct EncasedGpuSim {
    rotation: Vector2<f32>,
    translation: Vector2<f32>,
    scale: f32,
}

#[cfg(feature = "dim2")]
impl From<GpuSim2> for EncasedGpuSim {
    fn from(value: GpuSim2) -> Self {
        Self {
            rotation: [
                value.similarity.isometry.rotation.re,
                value.similarity.isometry.rotation.im,
            ]
            .into(),
            translation: value.similarity.isometry.translation.vector,
            scale: value.similarity.scaling(),
        }
    }
}

#[derive(Copy, Clone, Debug, ShaderType)]
#[cfg(feature = "dim3")]
struct EncasedGpuSim {
    rotation: Vector4<f32>,
    translation_scale: Vector4<f32>,
}

#[cfg(feature = "dim3")]
impl From<GpuSim3> for EncasedGpuSim {
    fn from(value: GpuSim3) -> Self {
        Self {
            rotation: value.isometry.rotation.coords,
            translation_scale: value.isometry.translation.vector.push(value.scaling()),
        }
    }
}

#[derive(Copy, Clone, Debug, ShaderType)]
struct GpuJointConstraintBuilder {
    body1: u32,
    body2: u32,
    joint_id: u32,
    joint: GpuGenericJoint,
    constraint_id: u32,
}

#[derive(Copy, Clone, Debug, ShaderType)]
struct GpuGenericJoint {
    local_frame_a: EncasedGpuSim,
    local_frame_b: EncasedGpuSim,
    locked_axes: u32,
    limit_axes: u32,
    motor_axes: u32,
    coupled_axes: u32,
    limits: [GpuJointLimits; SPATIAL_DIM],
    motors: [GpuJointMotor; SPATIAL_DIM],
}

impl From<GenericJoint> for GpuGenericJoint {
    fn from(value: GenericJoint) -> Self {
        Self {
            #[cfg(feature = "dim2")]
            local_frame_a: GpuSim::from(value.local_frame1).into(),
            #[cfg(feature = "dim2")]
            local_frame_b: GpuSim::from(value.local_frame2).into(),
            #[cfg(feature = "dim3")]
            local_frame_a: Similarity3::from_isometry(value.local_frame1, 1.0).into(),
            #[cfg(feature = "dim3")]
            local_frame_b: Similarity3::from_isometry(value.local_frame2, 1.0).into(),
            locked_axes: value.locked_axes.bits() as u32,
            limit_axes: value.limit_axes.bits() as u32,
            motor_axes: value.motor_axes.bits() as u32,
            coupled_axes: value.coupled_axes.bits() as u32,
            limits: value.limits.map(|e| e.into()),
            motors: value.motors.map(|e| e.into()),
        }
    }
}

#[derive(Copy, Clone, Debug, ShaderType)]
struct GpuJointLimits {
    min: f32,
    max: f32,
    impulse: f32,
}

impl From<JointLimits<f32>> for GpuJointLimits {
    fn from(value: JointLimits<f32>) -> Self {
        Self {
            min: value.min,
            max: value.max,
            impulse: value.impulse,
        }
    }
}

#[derive(Copy, Clone, Debug, ShaderType)]
struct GpuJointMotor {
    target_vel: f32,
    target_pos: f32,
    stiffness: f32,
    damping: f32,
    max_force: f32,
    impulse: f32,
    model: u32,
}

impl From<JointMotor> for GpuJointMotor {
    fn from(value: JointMotor) -> Self {
        Self {
            target_vel: value.target_vel,
            target_pos: value.target_pos,
            stiffness: value.stiffness,
            damping: value.damping,
            max_force: value.max_force,
            impulse: value.impulse,
            model: match value.model {
                MotorModel::AccelerationBased => 0,
                MotorModel::ForceBased => 1,
            },
        }
    }
}

#[derive(Copy, Clone, Debug, ShaderType)]
pub(crate) struct GpuJointConstraint {
    solver_vel_a: u32,
    solver_vel_b: u32,
    im_a: Vector<f32>,
    im_b: Vector<f32>,
    elements: [GpuJointConstraintElement; SPATIAL_DIM],
    len: u32,
}

#[derive(Copy, Clone, Debug, ShaderType)]
struct GpuJointConstraintElement {
    joint_id: u32,
    impulse: f32,
    impulse_bounds: Vector2<f32>,
    lin_jac: Vector<f32>,
    ang_jac_a: AngVector<f32>,
    ang_jac_b: AngVector<f32>,
    ii_ang_jac_a: AngVector<f32>,
    ii_ang_jac_b: AngVector<f32>,
    inv_lhs: f32,
    rhs: f32,
    rhs_wo_bias: f32,
    cfm_gain: f32,
    cfm_coeff: f32,
}

/// A set of impulse joints simulated on the GPU.
pub struct GpuImpulseJointSet {
    len: u32,
    num_colors: u32,
    max_color_group_len: u32,
    num_joints: GpuScalar<u32>,
    curr_color: GpuScalar<u32>,
    color_groups: GpuVector<u32>,
    joints: GpuVector<GpuImpulseJoint>,
    builders: GpuVector<GpuJointConstraintBuilder>,
    constraints: GpuVector<GpuJointConstraint>,
}

impl GpuImpulseJointSet {
    /// Converts a set of Rapier joints to a set of GPU joints.
    pub fn from_rapier(
        device: &Device,
        joints: &ImpulseJointSet,
        body_ids: &HashMap<RigidBodyHandle, u32>,
    ) -> Self {
        let usage = BufferUsages::STORAGE;
        let len = joints.len() as u32;
        let max_body_id = body_ids.values().copied().max().unwrap_or_default();

        // Convert joints.
        let mut unsorted_gpu_joints = vec![];
        for (_, joint) in joints.iter() {
            unsorted_gpu_joints.push(GpuImpulseJoint::from_rapier(joint, body_ids));
        }

        /*
         * Run a simple static greedy graph coloring, and group the joints.
         */
        let mut colors = vec![];
        let mut body_masks = vec![0u128; max_body_id as usize + 1];

        // Find colors.
        for joint in &unsorted_gpu_joints {
            // TODO: don’t take fixed bodies into account for the coloring.
            let a = joint.body_a as usize;
            let b = joint.body_b as usize;
            let mask = body_masks[a] | body_masks[b];
            let color = mask.trailing_ones();
            colors.push(color);
            body_masks[a] |= 1 << color;
            body_masks[b] |= 1 << color;
        }

        let num_colors = colors
            .iter()
            .copied()
            .max()
            .map(|n| n + 1)
            .unwrap_or_default();
        let mut color_groups = vec![0u32; num_colors as usize];

        // Count size of color groups.
        for color in &colors {
            color_groups[*color as usize] += 1;
        }

        let max_color_group_len = color_groups.iter().copied().max().unwrap_or_default();
        // println!(
        //     "Found {} colors. Max len: {}",
        //     num_colors, max_color_group_len
        // );

        // Prefix sum.
        for i in 0..color_groups.len().saturating_sub(1) {
            color_groups[i + 1] += color_groups[i];
        }

        // Bucket sort.
        let mut target = color_groups.clone();
        target.insert(0, 0);
        let mut sorted_gpu_joints = unsorted_gpu_joints.clone();

        for (joint, color) in unsorted_gpu_joints.iter().zip(colors.iter()) {
            sorted_gpu_joints[target[*color as usize] as usize] = *joint;
            target[*color as usize] += 1;
        }

        Self {
            len,
            num_colors,
            max_color_group_len,
            num_joints: GpuScalar::init(device, len, usage | BufferUsages::UNIFORM),
            curr_color: GpuScalar::init(device, 0, usage),
            color_groups: GpuVector::init(device, &color_groups, usage),
            joints: GpuVector::encase(device, &sorted_gpu_joints, usage),
            builders: GpuVector::uninit_encased(device, len, usage),
            constraints: GpuVector::uninit_encased(device, len, usage),
        }
    }

    /// Is this set empty?
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The number of joints in this set.
    pub fn len(&self) -> usize {
        self.len as usize
    }
}

#[derive(Shader)]
#[shader(
    derive(WgSim2, WgSim3),
    src = "joint.wgsl",
    src_fn = "substitute_aliases",
    shader_defs = "dim_shader_defs"
)]
/// Shader definition of joints.
pub struct WgJoint;

#[derive(Shader)]
#[shader(
    derive(WgJoint),
    src = "joint_constraint.wgsl",
    src_fn = "substitute_aliases",
    shader_defs = "dim_shader_defs"
)]
/// Shader definition of joint constraints.
pub struct WgJointConstraint;

#[derive(Shader)]
#[shader(
    derive(
        WgJoint,
        WgJointConstraint,
        WgSimParams,
        WgBody,
        WgSim2,
        WgSim3,
        WgQuat,
        WgRot2
    ),
    src = "joint_constraint_builder.wgsl",
    src_fn = "substitute_aliases",
    shader_defs = "dim_shader_defs"
)]
/// A solver responsible for initializing and solving impulse-based joint constraints.
pub struct WgJointSolver {
    init: ComputePipeline,
    update: ComputePipeline,
    solve: ComputePipeline,
    remove_bias: ComputePipeline,
    reset_color: ComputePipeline,
    inc_color: ComputePipeline,
}

/// Arguments given to the joint solver.
pub struct JointSolverArgs<'a> {
    /// The simulation parameters.
    pub sim_params: &'a GpuScalar<GpuSimParams>,
    /// The set of joints to solve.
    pub joints: &'a GpuImpulseJointSet,
    /// The body solvers.
    pub solver_vels: &'a GpuVector<GpuVelocity>,

    /// Rigid body poses.
    pub poses: &'a GpuVector<GpuSim>,
    /// World-space mass properties.
    pub mprops: &'a GpuVector<GpuWorldMassProperties>,
    /// Local-space mass properties.
    pub local_mprops: &'a GpuVector<GpuLocalMassProperties>,
}

impl WgJointSolver {
    const WORKGROUP_SIZE: u32 = 64;

    /// Generate joint constraints for this set of joint.
    pub fn init(&self, device: &Device, pass: &mut ComputePass, args: &JointSolverArgs) {
        KernelDispatch::new(device, pass, &self.init)
            .bind_at(
                0,
                [
                    (args.joints.num_joints.buffer(), 0),
                    (args.joints.joints.buffer(), 1),
                    (args.joints.builders.buffer(), 2),
                    (args.joints.constraints.buffer(), 3),
                ],
            )
            .bind_at(
                1,
                [(args.poses.buffer(), 1), (args.local_mprops.buffer(), 3)],
            )
            .dispatch(args.joints.len.div_ceil(Self::WORKGROUP_SIZE))
    }

    /// Updates the non-linear terms of the joint constraints.
    pub fn update(&self, device: &Device, pass: &mut ComputePass, args: &JointSolverArgs) {
        KernelDispatch::new(device, pass, &self.update)
            .bind_at(
                0,
                [
                    (args.joints.num_joints.buffer(), 0),
                    (args.joints.builders.buffer(), 2),
                    (args.joints.constraints.buffer(), 3),
                ],
            )
            .bind_at(
                1,
                [
                    (args.sim_params.buffer(), 0),
                    (args.poses.buffer(), 1),
                    (args.mprops.buffer(), 4),
                ],
            )
            .dispatch(args.joints.len.div_ceil(Self::WORKGROUP_SIZE))
    }

    /// Apply a single Projected-Gauss-Seidel step for solving joints.
    ///
    /// This intended to be used in the inner-loop of the TGS solver.
    pub fn solve(
        &self,
        device: &Device,
        pass: &mut ComputePass,
        args: &JointSolverArgs,
        use_bias: bool,
    ) {
        if !use_bias {
            KernelDispatch::new(device, pass, &self.remove_bias)
                .bind_at(
                    0,
                    [
                        (args.joints.num_joints.buffer(), 0),
                        (args.joints.constraints.buffer(), 3),
                    ],
                )
                .dispatch(args.joints.len.div_ceil(Self::WORKGROUP_SIZE))
        }

        KernelDispatch::new(device, pass, &self.reset_color)
            .bind_at(0, [(args.joints.curr_color.buffer(), 4)])
            .dispatch(1);
        for _ in 0..args.joints.num_colors {
            KernelDispatch::new(device, pass, &self.solve)
                .bind_at(
                    0,
                    [
                        (args.joints.constraints.buffer(), 3),
                        (args.joints.curr_color.buffer(), 4),
                        (args.joints.color_groups.buffer(), 5),
                    ],
                )
                .bind_at(1, [(args.solver_vels.buffer(), 2)])
                // TODO PERF: figure out a way to dispatch a number of threads that fits
                //            more tightly the size of the current color.
                .dispatch(
                    args.joints
                        .max_color_group_len
                        .div_ceil(Self::WORKGROUP_SIZE),
                );
            KernelDispatch::new(device, pass, &self.inc_color)
                .bind_at(0, [(args.joints.curr_color.buffer(), 4)])
                .dispatch(1);
        }
    }
}

wgcore::test_shader_compilation!(WgJoint, wgcore, wgparry::dim_shader_defs());
wgcore::test_shader_compilation!(WgJointConstraint, wgcore, wgparry::dim_shader_defs());
wgcore::test_shader_compilation!(WgJointSolver, wgcore, wgparry::dim_shader_defs());
