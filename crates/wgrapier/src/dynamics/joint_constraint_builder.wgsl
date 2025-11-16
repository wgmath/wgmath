#define_import_path wgrapier::dynamics::joint_constraint_builder

#import wgrapier::body as Body;
#import wgrapier::dynamics::sim_params as Params;
#import wgrapier::dynamics::joint as Joint
#import wgrapier::dynamics::joint_constraint as JointConstraint

#if DIM == 2
    #import wgebra::sim2 as Pose
    #import wgebra::rot2 as Rot
const DIM: u32 = 2;
#else
    #import wgebra::sim3 as Pose
    #import wgebra::quat as Rot
const DIM: u32 = 3;
#endif

const MAX: f32 = 1.0e20;

@group(0) @binding(0)
var<uniform> joints_len: u32;
@group(0) @binding(1)
var<storage, read> joints: array<Joint::ImpulseJoint>;
@group(0) @binding(2)
var<storage, read_write> builders: array<JointConstraintBuilder>;
@group(0) @binding(3)
var<storage, read_write> constraints: array<JointConstraint::JointConstraint>;
@group(0) @binding(4)
var<storage, read_write> curr_color: u32;
@group(0) @binding(5)
var<storage, read> color_groups: array<u32>;


@group(1) @binding(0)
var<uniform> params: Params::SimParams;
@group(1) @binding(1)
var<storage, read_write> poses: array<Transform>;
@group(1) @binding(2)
var<storage, read_write> solver_vels: array<Body::Velocity>;
@group(1) @binding(3)
var<storage, read> local_mprops: array<Body::LocalMassProperties>;
@group(1) @binding(4)
var<storage, read> mprops: array<Body::WorldMassProperties>;

const WORKGROUP_SIZE: u32 = 64;

@compute @workgroup_size(1, 1, 1)
fn reset_color() {
    // NOTE: for joints, our first colors start at 0.
    curr_color = 0u;
}

@compute @workgroup_size(1, 1, 1)
fn inc_color() {
    curr_color += 1;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn init(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE * num_workgroups.y * num_workgroups.z;
    for (var i = invocation_id.x; i < joints_len; i += num_threads) {
        init_builder_and_constraint(i, i, i); // TODO: will these three indices ever be different?
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn update(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE * num_workgroups.y * num_workgroups.z;
    for (var i = invocation_id.x; i < joints_len; i += num_threads) {
        update_constraint(i, i); // TODO: will these two indices ever be different?
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn remove_bias(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE * num_workgroups.y * num_workgroups.z;
    for (var i = invocation_id.x; i < joints_len; i += num_threads) {
        for (var j = 0u; j < constraints[i].len; j++) {
            constraints[i].elements[j].rhs = constraints[i].elements[j].rhs_wo_bias;
        }
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn solve(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE * num_workgroups.y * num_workgroups.z;
    var start = 0u;
    let end = color_groups[curr_color];

    if curr_color > 0u {
        start = color_groups[curr_color - 1u];
    }

    for (var i = start + invocation_id.x; i < end; i += num_threads) {
        solve_constraint(i); // TODO: will these three indices ever be different?
    }
}

struct JointConstraintBuilder {
    body1: u32,
    body2: u32,
    joint_id: u32,
    joint: Joint::GenericJoint,
    constraint_id: u32,
}

fn init_builder_and_constraint(
    joint_id: u32,
    out_builder_id: u32,
    constraint_id: u32,
) {
    let body1 = joints[joint_id].body_a;
    let body2 = joints[joint_id].body_b;

    var joint_data = joints[joint_id].data;

    // TODO: if we switch to solver body poses given in center-of-mass space,
    // we need to transform the anchors to that space.
    if all(local_mprops[body1].inv_mass == Vector(0.0)) && false {
        joint_data.local_frame_a = Pose::mul(poses[body1], joint_data.local_frame_a);
    } else {
        #if DIM == 2
        joint_data.local_frame_a.translation -= local_mprops[body1].com;
        #else
        joint_data.local_frame_a.translation_scale -= vec4(local_mprops[body1].com, 0.0);
        #endif
    }

    if all(local_mprops[body2].inv_mass == Vector(0.0)) && false {
        joint_data.local_frame_b = Pose::mul(poses[body2], joint_data.local_frame_b);
    } else {
        #if DIM == 2
        joint_data.local_frame_b.translation -= local_mprops[body2].com;
        #else
        joint_data.local_frame_b.translation_scale -= vec4(local_mprops[body2].com, 0.0);
        #endif
    }

    builders[out_builder_id] = JointConstraintBuilder(
        body1,
        body2,
        joint_id,
        joint_data,
        constraint_id,
    );
    constraints[constraint_id].solver_vel_a = body1;
    constraints[constraint_id].solver_vel_b = body2;
    constraints[constraint_id].im_a = local_mprops[body1].inv_mass;
    constraints[constraint_id].im_b = local_mprops[body2].inv_mass;
    constraints[constraint_id].len = 0; // Constraint elements will be filled later.
}

fn update_constraint(
    builder_id: u32,
    constraint_id: u32,
) {
    // NOTE: right now, the "update", is basically reconstructing all the
    //       constraints entirely. Could we make this more incremental?
    let joint = &builders[builder_id].joint;
    let body1 = builders[builder_id].body1;
    let body2 = builders[builder_id].body2;
    let pose1 = poses[body1];
    let pose2 = poses[body2];
    let mprops1 = mprops[body1];
    let mprops2 = mprops[body2];

    let frame1 = Pose::mul(pose1, joint.local_frame_a);
    let frame2 = Pose::mul(pose2, joint.local_frame_b);

    // TODO: needs adjustment if the pose origin isn’t the same as the
    //       center of mass.
    #if DIM == 2
    let world_com1 = pose1.translation;
    let world_com2 = pose2.translation;
    #else
    let world_com1 = pose1.translation_scale.xyz;
    let world_com2 = pose2.translation_scale.xyz;
    #endif

    let joint_body1 = JointConstraint::JointSolverBody(
        mprops1.inv_mass,
        mprops1.inv_inertia,
        world_com1,
        body1,
    );
    let joint_body2 = JointConstraint::JointSolverBody(
        mprops2.inv_mass,
        mprops2.inv_inertia,
        world_com2,
        body2,
    );

    var len = 0u;
    let locked_axes = joint.locked_axes;
    let motor_axes = joint.motor_axes & ~locked_axes;
    let limit_axes = joint.limit_axes & ~locked_axes;
    let coupled_axes = joint.coupled_axes;

    // The has_lin/ang_coupling test is needed to avoid shl overflow later.
    let has_lin_coupling = (coupled_axes & Joint::LIN_AXES_MASK) != 0;
    let first_coupled_lin_axis_id =
        countTrailingZeros(coupled_axes & Joint::LIN_AXES_MASK);

#if DIM == 3
    let has_ang_coupling = (coupled_axes & Joint::ANG_AXES_MASK) != 0;
    let first_coupled_ang_axis_id =
        countTrailingZeros(coupled_axes & Joint::ANG_AXES_MASK);
#endif

    let helper = new_helper(
        frame1,
        frame2,
        mprops1.com,
        mprops2.com,
        locked_axes,
    );

    var start = len;
    for (var i = DIM; i < Joint::SPATIAL_DIM; i++) {
        if ((motor_axes & ~coupled_axes) & (1u << i)) != 0u {
            let motor_params = Joint::motor_params(joint.motors[i], params.dt);
            constraints[constraint_id].elements[len] = motor_angular(
                helper,
                constraint_id, // TODO: is this really useful?
                joint_body1,
                joint_body2,
                i - DIM,
                motor_params,
            );
            len += 1u;
        }
    }

    for (var i = 0u; i < DIM; i++) {
        if ((motor_axes & ~coupled_axes) & (1u << i)) != 0u {
            var limits = vec2(-MAX, MAX);

            if (limit_axes & (1u << i)) != 0u {
               limits = vec2(joint.limits[i].min, joint.limits[i].max);
            }

            let motor_params = Joint::motor_params(joint.motors[i], params.dt);

            constraints[constraint_id].elements[len] = motor_linear(
                helper,
                constraint_id,
                joint_body1,
                joint_body2,
                i,
                motor_params,
                limits,
            );
            len += 1u;
        }
    }

    if ((motor_axes & coupled_axes) & Joint::ANG_AXES_MASK) != 0u {
        // TODO: coupled angular motor constraint.
    }

    if ((motor_axes & coupled_axes) & Joint::LIN_AXES_MASK) != 0u {
        var limits = vec2(-MAX, MAX);
        if (limit_axes & (1u << first_coupled_lin_axis_id)) != 0u {
            limits = vec2(
                joint.limits[first_coupled_lin_axis_id].min,
                joint.limits[first_coupled_lin_axis_id].max,
            );
        }

        let motor_params = Joint::motor_params(joint.motors[first_coupled_lin_axis_id], params.dt);

        constraints[constraint_id].elements[len] = motor_linear_coupled(
            helper,
            constraint_id,
            joint_body1,
            joint_body2,
            coupled_axes,
            motor_params,
            limits,
        );
        len += 1u;
    }

    orthogonalize_constraints(constraint_id, start, len);

    start = len;
    for (var i = DIM; i < Joint::SPATIAL_DIM; i++) {
        if (locked_axes & (1u << i)) != 0u {
            constraints[constraint_id].elements[len] = lock_angular(
                helper,
                constraint_id,
                joint_body1,
                joint_body2,
                i - DIM,
            );
            len += 1u;
        }
    }
    for (var i = 0u; i < DIM; i++) {
        if (locked_axes & (1u << i)) != 0u {
            constraints[constraint_id].elements[len] =
                lock_linear(helper, constraint_id, joint_body1, joint_body2, i);
            len += 1u;
        }
    }

    for (var i = DIM; i < Joint::SPATIAL_DIM; i++) {
        if ((limit_axes & ~coupled_axes) & (1u << i)) != 0u {
            constraints[constraint_id].elements[len] = limit_angular(
                helper,
                constraint_id,
                joint_body1,
                joint_body2,
                i - DIM,
                vec2(joint.limits[i].min, joint.limits[i].max),
            );
            len += 1u;
        }
    }
    for (var i = 0u; i < DIM; i++) {
        if ((limit_axes & ~coupled_axes) & (1u << i)) != 0u {
            constraints[constraint_id].elements[len] = limit_linear(
                helper,
                constraint_id,
                joint_body1,
                joint_body2,
                i,
                vec2(joint.limits[i].min, joint.limits[i].max),
            );
            len += 1u;
        }
    }

//    #[cfg(feature = "dim3")]
//    if has_ang_coupling && (limit_axes & (1 << first_coupled_ang_axis_id)) != 0 {
//        constraints[constraint_id].elements[len] = helper.limit_angular_coupled(
//            constraint_id,
//            joint_body1,
//            joint_body2,
//            coupled_axes,
//            [
//                joint.limits[first_coupled_ang_axis_id].min,
//                joint.limits[first_coupled_ang_axis_id].max,
//            ],
//            WritebackId::Limit(first_coupled_ang_axis_id),
//        );
//        len += 1;
//    }

    if has_lin_coupling && (limit_axes & (1u << first_coupled_lin_axis_id)) != 0u {
        constraints[constraint_id].elements[len] = limit_linear_coupled(
            helper,
            constraint_id,
            joint_body1,
            joint_body2,
            coupled_axes,
            vec2(
                joint.limits[first_coupled_lin_axis_id].min,
                joint.limits[first_coupled_lin_axis_id].max,
            ),
        );
        len += 1u;
    }

    orthogonalize_constraints(constraint_id, start, len);
    constraints[constraint_id].len = len;
}


struct JointConstraintHelper {
#if DIM == 2
    basis: mat2x2<f32>,
    cmat1_basis: vec2<f32>,
    cmat2_basis: vec2<f32>,
    lin_err: Vector,
    ang_err: Rot::Rot2
#else
    basis: mat3x3<f32>,
    basis2: mat3x3<f32>, // TODO: used only for angular coupling. Can we avoid storing this?
    cmat1_basis: mat3x3<f32>,
    cmat2_basis: mat3x3<f32>,
    ang_basis: mat3x3<f32>,
    lin_err: Vector,
    ang_err: Rot::Quat
#endif
}

fn new_helper(
    frame1_: Transform,
    frame2: Transform,
    world_com1: Vector,
    world_com2: Vector,
    locked_lin_axes: u32,
) -> JointConstraintHelper {
#if DIM == 2
    var frame1 = frame1_;
    let basis = Rot::toMatrix(frame1.rotation);
    let lin_err = frame2.translation - frame1.translation;

    // Adjust the point of application of the force for the first body,
    // by snapping free axes to the second frame’s center (to account for
    // the allowed relative movement).
    {
        var new_center1 = frame2.translation; // First, assume all dofs are free.

        // Then snap the locked ones.
        for (var i = 0u; i < DIM; i++) {
            if (locked_lin_axes & (1u << i)) != 0u {
                let axis = basis[i];
                new_center1 -= axis * dot(lin_err, axis);
            }
        }
        frame1.translation = new_center1;
    }

    let r1 = frame1.translation - world_com1;
    let r2 = frame2.translation - world_com2;

    let cmat1 = gcross_matrix(r1);
    let cmat2 = gcross_matrix(r2);

    var ang_err = Rot::mul(Rot::inv(frame1.rotation), frame2.rotation);

    return JointConstraintHelper(
        basis,
        cmat1 * basis,
        cmat2 * basis,
        lin_err,
        ang_err,
    );
#else
    var frame1 = frame1_;
    let basis = Rot::toMatrix(frame1.rotation);
    let lin_err = frame2.translation_scale.xyz - frame1.translation_scale.xyz;

    // Adjust the point of application of the force for the first body,
    // by snapping free axes to the second frame’s center (to account for
    // the allowed relative movement).
    {
        var new_center1 = frame2.translation_scale.xyz; // First, assume all dofs are free.

        // Then snap the locked ones.
        for (var i = 0u; i < DIM; i++) {
            if (locked_lin_axes & (1u << i)) != 0u {
                let axis = basis[i];
                new_center1 -= axis * dot(lin_err, axis);
            }
        }
        frame1.translation_scale = vec4(new_center1, frame1.translation_scale.w);
    }

    let r1 = frame1.translation_scale.xyz - world_com1;
    let r2 = frame2.translation_scale.xyz - world_com2;

    let cmat1 = gcross_matrix(r1);
    let cmat2 = gcross_matrix(r2);

    var ang_basis = transpose(Rot::diff_conj1_2(frame1.rotation, frame2.rotation));
    var ang_err = Rot::mul(Rot::inv(frame1.rotation), frame2.rotation);
    let sgn = select(-1.0, 1.0, dot(frame1.rotation.coords, frame2.rotation.coords) > 0.0);
    ang_basis *= sgn;
    ang_err.coords *= sgn;

    return JointConstraintHelper(
        basis,
        Rot::toMatrix(frame2.rotation),
        cmat1 * basis,
        cmat2 * basis,
        ang_basis,
        lin_err,
        ang_err,
    );
#endif
}

fn limit_linear(
    helper: JointConstraintHelper,
    joint_id: u32,
    body1: JointConstraint::JointSolverBody,
    body2: JointConstraint::JointSolverBody,
    limited_axis: u32,
    limits: vec2<f32>,
) -> JointConstraint::JointConstraintElement {
    var constraint =
        lock_linear(helper, joint_id, body1, body2, limited_axis);

    let dist = dot(helper.lin_err, constraint.lin_jac);
    let min_enabled = dist <= limits[0];
    let max_enabled = limits[1] <= dist;

    let erp_inv_dt = Params::joint_erp_inv_dt(params);
    let cfm_coeff = Params::joint_cfm_coeff(params);
    let rhs_bias =
        (max(dist - limits[1], 0.0) - max(limits[0] - dist, 0.0)) * erp_inv_dt;
    constraint.rhs = constraint.rhs_wo_bias + rhs_bias;
    constraint.cfm_coeff = cfm_coeff;
    constraint.impulse_bounds = vec2(
        select(0.0, -MAX, min_enabled),
        select(0.0, MAX, max_enabled),
    );

    return constraint;
}

fn limit_linear_coupled(
    helper: JointConstraintHelper,
    joint_id: u32,
    body1: JointConstraint::JointSolverBody,
    body2: JointConstraint::JointSolverBody,
    coupled_axes: u32,
    limits: vec2<f32>,
) -> JointConstraint::JointConstraintElement {
    var lin_jac = Vector(0.0);
    var ang_jac1 = AngVector(0.0);
    var ang_jac2 = AngVector(0.0);

    for (var i = 0u; i < DIM; i++) {
        if (coupled_axes & (1u << i)) != 0u {
            let coeff = dot(helper.basis[i], helper.lin_err);
            lin_jac += helper.basis[i] * coeff;
#if DIM == 2
            ang_jac1 += helper.cmat1_basis[i] * coeff;
            ang_jac2 += helper.cmat2_basis[i] * coeff;
#else
            ang_jac1 += helper.cmat1_basis[i] * coeff;
            ang_jac2 += helper.cmat2_basis[i] * coeff;
#endif
        }
    }

    // FIXME: handle min limit too.

    let dist = length(lin_jac);
    let inv_dist = pseudo_inv(dist);
    lin_jac *= inv_dist;
    ang_jac1 *= inv_dist;
    ang_jac2 *= inv_dist;

    let rhs_wo_bias = min(dist - limits[1], 0.0) * Params::inv_dt(params);

    let ii_ang_jac1 = body1.ii * ang_jac1;
    let ii_ang_jac2 = body2.ii * ang_jac2;

    let erp_inv_dt = Params::joint_erp_inv_dt(params);
    let cfm_coeff = Params::joint_cfm_coeff(params);
    let rhs_bias = max(dist - limits[1], 0.0) * erp_inv_dt;
    let rhs = rhs_wo_bias + rhs_bias;
    let impulse_bounds = vec2(0.0, MAX);

    return JointConstraint::JointConstraintElement(
        joint_id,
        0.0, // impulse
        impulse_bounds,
        lin_jac,
        ang_jac1,
        ang_jac2,
        ii_ang_jac1,
        ii_ang_jac2,
        0.0, // inv_lhs will be set during orthogonalization.
        rhs,
        rhs_wo_bias,
        0.0, // cfm_gain
        cfm_coeff,
    );
}

fn motor_linear(
    helper: JointConstraintHelper,
    joint_id: u32,
    body1: JointConstraint::JointSolverBody,
    body2: JointConstraint::JointSolverBody,
    motor_axis: u32,
    motor_params: Joint::MotorParameters,
    limits: vec2<f32>,
) -> JointConstraint::JointConstraintElement {
    let inv_dt = Params::inv_dt(params);
    var constraint =
        lock_linear(helper, joint_id, body1, body2, motor_axis);

    var rhs_wo_bias = 0.0;
    if motor_params.erp_inv_dt != 0.0 {
        let dist = dot(helper.lin_err, constraint.lin_jac);
        rhs_wo_bias += (dist - motor_params.target_pos) * motor_params.erp_inv_dt;
    }

    var target_vel = motor_params.target_vel;
    if any(limits != vec2(-MAX, MAX)) {
        let dist = dot(helper.lin_err, constraint.lin_jac);
        target_vel =
            clamp(target_vel, (limits[0] - dist) * inv_dt, (limits[1] - dist) * inv_dt);
    };

    rhs_wo_bias += -target_vel;

    constraint.cfm_coeff = motor_params.cfm_coeff;
    constraint.cfm_gain = motor_params.cfm_gain;
    constraint.impulse_bounds = vec2(-motor_params.max_impulse, motor_params.max_impulse);
    constraint.rhs = rhs_wo_bias;
    constraint.rhs_wo_bias = rhs_wo_bias;
    return constraint;
}

fn motor_linear_coupled(
    helper: JointConstraintHelper,
    joint_id: u32,
    body1: JointConstraint::JointSolverBody,
    body2: JointConstraint::JointSolverBody,
    coupled_axes: u32,
    motor_params: Joint::MotorParameters,
    limits: vec2<f32>,
) -> JointConstraint::JointConstraintElement {
    let inv_dt = Params::inv_dt(params);

    var lin_jac = Vector(0.0);
    var ang_jac1 = AngVector(0.0);
    var ang_jac2 = AngVector(0.0);

    for (var i = 0u; i < DIM; i++) {
        if (coupled_axes & (1u << i)) != 0u {
            let coeff = dot(helper.basis[i], helper.lin_err);
            lin_jac += helper.basis[i] * coeff;
#if DIM == 2
            ang_jac1 += helper.cmat1_basis[i] * coeff;
            ang_jac2 += helper.cmat2_basis[i] * coeff;
#else
            ang_jac1 += helper.cmat1_basis[i] * coeff;
            ang_jac2 += helper.cmat2_basis[i] * coeff;
#endif
        }
    }

    let dist = length(lin_jac);
    let inv_dist = pseudo_inv(dist);
    lin_jac *= inv_dist;
    ang_jac1 *= inv_dist;
    ang_jac2 *= inv_dist;

    var rhs_wo_bias = 0.0;
    if motor_params.erp_inv_dt != 0.0 {
        rhs_wo_bias += (dist - motor_params.target_pos) * motor_params.erp_inv_dt;
    }

    var target_vel = motor_params.target_vel;
    if any(limits != vec2(-MAX, MAX)) {
        target_vel =
            clamp(target_vel, (limits[0] - dist) * inv_dt, (limits[1] - dist) * inv_dt);
    };

    rhs_wo_bias += -target_vel;

    let ii_ang_jac1 = body1.ii * ang_jac1;
    let ii_ang_jac2 = body2.ii * ang_jac2;

    return JointConstraint::JointConstraintElement(
        joint_id,
        0.0, // impulse
        vec2(-motor_params.max_impulse, motor_params.max_impulse),
        lin_jac,
        ang_jac1,
        ang_jac2,
        ii_ang_jac1,
        ii_ang_jac2,
        0.0, // inv_lhs will be set during orthogonalization.
        rhs_wo_bias,
        rhs_wo_bias,
        motor_params.cfm_gain,
        motor_params.cfm_coeff,
    );
}

fn lock_linear(
    helper: JointConstraintHelper,
    joint_id: u32,
    body1: JointConstraint::JointSolverBody,
    body2: JointConstraint::JointSolverBody,
    locked_axis: u32,
) -> JointConstraint::JointConstraintElement {
    let lin_jac = helper.basis[locked_axis];
    let ang_jac1 = helper.cmat1_basis[locked_axis];
    let ang_jac2 = helper.cmat2_basis[locked_axis];

    let rhs_wo_bias = 0.0;
    let erp_inv_dt = Params::joint_erp_inv_dt(params);
    let cfm_coeff = Params::joint_cfm_coeff(params);
    let rhs_bias = dot(lin_jac, helper.lin_err) * erp_inv_dt;

    let ii_ang_jac1 = body1.ii * ang_jac1;
    let ii_ang_jac2 = body2.ii * ang_jac2;

    return JointConstraint::JointConstraintElement(
        joint_id,
        0.0,             // impulse
        vec2(-MAX, MAX), // impulse_bounds
        lin_jac,         // lin_jac
        ang_jac1,
        ang_jac2,
        ii_ang_jac1,
        ii_ang_jac2,
        0.0,             // inv_lhs will be set during orthogonalization.
        rhs_wo_bias + rhs_bias,
        rhs_wo_bias,
        0.0,             // cfm_gain
        cfm_coeff,
    );
}

fn limit_angular(
    helper: JointConstraintHelper,
    joint_id: u32,
    body1: JointConstraint::JointSolverBody,
    body2: JointConstraint::JointSolverBody,
    limited_axis: u32,
    limits: vec2<f32>,
) -> JointConstraint::JointConstraintElement {
    let s_limits = vec2(sin(limits[0] * 0.5), sin(limits[1] * 0.5));
#if DIM == 2
    let s_ang = sin(Rot::angle(helper.ang_err) * 0.5);
#else
    let s_ang = Rot::imag(helper.ang_err)[limited_axis];
#endif
    let min_enabled = s_ang <= s_limits[0];
    let max_enabled = s_limits[1] <= s_ang;

    let impulse_bounds = vec2(
        select(0.0, -MAX, min_enabled),
        select(0.0, MAX, max_enabled),
    );

#if DIM == 2
    let ang_jac = 1.0;
#else
    let ang_jac = helper.ang_basis[limited_axis];
#endif
    let rhs_wo_bias = 0.0;
    let erp_inv_dt = Params::joint_erp_inv_dt(params);
    let cfm_coeff = Params::joint_cfm_coeff(params);
    let rhs_bias = (max(s_ang - s_limits[1], 0.0)
        - max(s_limits[0] - s_ang, 0.0))
        * erp_inv_dt;

    let ii_ang_jac1 = body1.ii * ang_jac;
    let ii_ang_jac2 = body2.ii * ang_jac;

    return JointConstraint::JointConstraintElement(
        joint_id,
        0.0,            // impulse
        impulse_bounds,
        Vector(),       // lin_jac
        ang_jac,
        ang_jac,
        ii_ang_jac1,
        ii_ang_jac2,
        0.0,            // inv_lhs will be set during orthogonalization.
        rhs_wo_bias + rhs_bias,
        rhs_wo_bias,
        0.0,            // cfm_gain
        cfm_coeff,
    );
}

fn motor_angular(
    helper: JointConstraintHelper,
    joint_id: u32,
    body1: JointConstraint::JointSolverBody,
    body2: JointConstraint::JointSolverBody,
    motor_axis: u32,
    motor_params: Joint::MotorParameters,
) -> JointConstraint::JointConstraintElement {
#if DIM == 2
    let ang_jac = 1.0;
#else
    let ang_jac = helper.basis[motor_axis];
#endif

    var rhs_wo_bias = 0.0;
    if motor_params.erp_inv_dt != 0.0 {
#if DIM == 2
        let ang_dist = Rot::angle(helper.ang_err);
#else
        // Clamp the component from -1.0 to 1.0 to account for slight imprecision
        let clamped_err = clamp(Rot::imag(helper.ang_err)[motor_axis], -1.0, 1.0);
        let ang_dist = asin(clamped_err) * 2.0;
#endif

        let target_ang = motor_params.target_pos;
        rhs_wo_bias += smallest_abs_diff_between_angles(ang_dist, target_ang)
            * motor_params.erp_inv_dt;
    }

    rhs_wo_bias += -motor_params.target_vel;

    let ii_ang_jac1 = body1.ii * ang_jac;
    let ii_ang_jac2 = body2.ii * ang_jac;

    return JointConstraint::JointConstraintElement(
        joint_id,
        0.0,         // impulse
        vec2(-motor_params.max_impulse, motor_params.max_impulse), // impulse_bounds
        Vector(),    // lin_jac
        ang_jac,
        ang_jac,
        ii_ang_jac1,
        ii_ang_jac2,
        0.0,         // inv_lhs will be set during orthogonalization.
        rhs_wo_bias, // rhs
        rhs_wo_bias,
        motor_params.cfm_gain,
        motor_params.cfm_coeff,
    );
}

fn lock_angular(
    helper: JointConstraintHelper,
    joint_id: u32,
    body1: JointConstraint::JointSolverBody,
    body2: JointConstraint::JointSolverBody,
    locked_axis: u32,
) -> JointConstraint::JointConstraintElement {
#if DIM == 2
    let ang_jac = 1.0;
#else
    let ang_jac = helper.ang_basis[locked_axis];
#endif

    let rhs_wo_bias = 0.0;
    let erp_inv_dt = Params::joint_erp_inv_dt(params);
    let cfm_coeff = Params::joint_cfm_coeff(params);
#if DIM == 2
    let rhs_bias = helper.ang_err.cos_sin.y * erp_inv_dt;
#else
    let rhs_bias = Rot::imag(helper.ang_err)[locked_axis] * erp_inv_dt;
#endif
    let ii_ang_jac1 = body1.ii * ang_jac;
    let ii_ang_jac2 = body2.ii * ang_jac;

    return JointConstraint::JointConstraintElement(
        joint_id,
        0.0,             // impulse
        vec2(-MAX, MAX), // impulse_bounds
        Vector(),        // lin_jac
        ang_jac,
        ang_jac,
        ii_ang_jac1,
        ii_ang_jac2,
        0.0,             // inv_lhs will be set during orthogonalization.
        rhs_wo_bias + rhs_bias,
        rhs_wo_bias,
        0.0,             // cfm_gain
        cfm_coeff,
    );
}

/// Orthogonalize the constraints and set their inv_lhs field.
fn orthogonalize_constraints(id: u32, start: u32, end: u32) {
    let len = end - start;

    if len == 0 {
        return;
    }

    let imsum = constraints[id].im_a + constraints[id].im_b;

    // Use the modified Gram-Schmidt orthogonalization.
    for (var j = start; j < end; j++) {
        let dot_jj = dot(constraints[id].elements[j].lin_jac, imsum * constraints[id].elements[j].lin_jac)
            + gdot(constraints[id].elements[j].ii_ang_jac_a, constraints[id].elements[j].ang_jac_a)
            + gdot(constraints[id].elements[j].ii_ang_jac_b, constraints[id].elements[j].ang_jac_b);
        let cfm_gain = dot_jj * constraints[id].elements[j].cfm_coeff + constraints[id].elements[j].cfm_gain;
        let inv_dot_jj = pseudo_inv(dot_jj);
        constraints[id].elements[j].inv_lhs = pseudo_inv(dot_jj + cfm_gain); // Don’t forget to update the inv_lhs.
        constraints[id].elements[j].cfm_gain = cfm_gain;

        if any(constraints[id].elements[j].impulse_bounds != vec2(-MAX, MAX)) {
            // Don't remove constraints with limited forces from the others
            // because they may not deliver the necessary forces to fulfill
            // the removed parts of other constraints.
            continue;
        }

        for (var i = j + 1u; i < end; i++) {
            let dot_ij = dot(constraints[id].elements[i].lin_jac, imsum * constraints[id].elements[j].lin_jac)
                + gdot(constraints[id].elements[i].ii_ang_jac_a, constraints[id].elements[j].ang_jac_a)
                + gdot(constraints[id].elements[i].ii_ang_jac_b, constraints[id].elements[j].ang_jac_b);
            let coeff = dot_ij * inv_dot_jj;

            constraints[id].elements[i].lin_jac -= constraints[id].elements[j].lin_jac * coeff;
            constraints[id].elements[i].ang_jac_a -= constraints[id].elements[j].ang_jac_a * coeff;
            constraints[id].elements[i].ang_jac_b -= constraints[id].elements[j].ang_jac_b * coeff;
            constraints[id].elements[i].ii_ang_jac_a -= constraints[id].elements[j].ii_ang_jac_a * coeff;
            constraints[id].elements[i].ii_ang_jac_b -= constraints[id].elements[j].ii_ang_jac_b * coeff;
            constraints[id].elements[i].rhs_wo_bias -= constraints[id].elements[j].rhs_wo_bias * coeff;
            constraints[id].elements[i].rhs -= constraints[id].elements[j].rhs * coeff;
        }
    }
}

/*
fn limit_angular_coupled(
    helper: JointConstraintHelper,
    joint_id: u32,
    body1: JointConstraint::JointSolverBody,
    body2: JointConstraint::JointSolverBody,
    coupled_axes: u32,
    limits: vec2<f32>,
) -> JointConstraint::JointConstraintElement {
    // NOTE: right now, this only supports exactly 2 coupled axes.
    let ang_coupled_axes = coupled_axes >> DIM;
    let not_coupled_index = ang_coupled_axes.trailing_ones() as u32;
    let axis1 = helper.basis[not_coupled_index];
    let axis2 = helper.basis2[not_coupled_index];

    let rot = Rotation::rotation_between(&axis1, &axis2).unwrap_or_else(Rotation::identity);
    let (ang_jac, angle) = rot
        .axis_angle()
        .map(|(axis, angle)| (axis.into_inner(), angle))
        .unwrap_or_else(|| (axis1.orthonormal_basis()[0], 0.0));

    let min_enabled = angle <= limits[0];
    let max_enabled = limits[1] <= angle;

    let impulse_bounds = vec2(
        select(0.0, -MAX, min_enabled),
        select(0.0, MAX, max_enabled),
    );

    let rhs_wo_bias = 0.0;

    let erp_inv_dt = Params::joint_erp_inv_dt(params);
    let cfm_coeff = Params::joint_cfm_coeff(params);
    let rhs_bias = (max(angle - limits[1], 0.0) - max(limits[0] - angle, 0.0)) * erp_inv_dt;

    let ii_ang_jac1 = body1.ii * ang_jac;
    let ii_ang_jac2 = body2.ii * ang_jac;

    return JointConstraint::JointConstraintElement(
        joint_id,
        0.0,            // impulse
        impulse_bounds,
        Vector(),       // lin_jac
        ang_jac,
        ang_jac,
        ii_ang_jac1,
        ii_ang_jac2,
        0.0,            // inv_lhs will be set during orthogonalization.
        rhs_wo_bias + rhs_bias,
        rhs_wo_bias,
        0.0,            // cfm_gain
        cfm_coeff,
    );
}
*/

fn solve_constraint(constraint_id: u32) {
    let constraint = &constraints[constraint_id];
    var solver_vel1 = solver_vels[constraint.solver_vel_a];
    var solver_vel2 = solver_vels[constraint.solver_vel_b];

    for (var i = 0u; i < constraint.len; i++) {
        let element = &constraint.elements[i];
        let dlinvel = dot(element.lin_jac, solver_vel2.linear - solver_vel1.linear);
        let dangvel =
            gdot(element.ang_jac_b, solver_vel2.angular) - gdot(element.ang_jac_a, solver_vel1.angular);

        let rhs = dlinvel + dangvel + element.rhs;
        let total_impulse = clamp(element.impulse + element.inv_lhs * (rhs - element.cfm_gain * element.impulse),
            element.impulse_bounds[0], element.impulse_bounds[1]);
        let delta_impulse = total_impulse - element.impulse;
        element.impulse = total_impulse;

        let lin_impulse = element.lin_jac * delta_impulse;

        solver_vel1.linear += lin_impulse * constraint.im_a;
        solver_vel1.angular += element.ii_ang_jac_a * delta_impulse;
        solver_vel2.linear -= lin_impulse * constraint.im_b;
        solver_vel2.angular -= element.ii_ang_jac_b * delta_impulse;
    }

    solver_vels[constraint.solver_vel_a] = solver_vel1;
    solver_vels[constraint.solver_vel_b] = solver_vel2;
}

#if DIM ==  2
fn gcross_matrix(r: vec2<f32>) -> vec2<f32> {
    return vec2(-r.y, r.x);
}
#else
fn gcross_matrix(r: vec3<f32>) -> mat3x3<f32> {
    return mat3x3(
        vec3(0.0, r.z, -r.y),
        vec3(-r.z, 0.0, r.x),
        vec3(r.y, -r.x, 0.0),
    );
}
#endif

fn smallest_abs_diff_between_angles(a: f32, b: f32) -> f32 {
    // Select the smallest path among the two angles to reach the target.
    let s_err = a - b;
    let sgn = sign(s_err);
    let s_err_complement = s_err - sgn * Params::TWO_PI;
    let s_err_is_smallest = abs(s_err) < abs(s_err_complement);
    return select(s_err_complement, s_err, s_err_is_smallest);
}

fn pseudo_inv(x: f32) -> f32 {
    return select(1.0 / x, 0.0, x == 0.0);
}

#if DIM == 2
fn gdot(a: f32, b: f32) -> f32 {
    return a * b;
}
#else
fn gdot(a: vec3<f32>, b: vec3<f32>) -> f32 {
    return dot(a, b);
}
#endif