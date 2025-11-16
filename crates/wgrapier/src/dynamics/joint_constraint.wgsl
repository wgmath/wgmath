#define_import_path wgrapier::dynamics::joint_constraint

#import wgrapier::dynamics::joint as Joint

struct MotorParameters {
    erp_inv_dt: f32,
    cfm_coeff: f32,
    cfm_gain: f32,
    target_pos: f32,
    target_vel: f32,
    max_impulse: f32,
}

struct JointSolverBody {
    im: Vector,
#if DIM == 2
    ii: f32,
#else
    ii: mat3x3<f32>,
#endif
    // TODO: is this still needed now that the solver body poses are expressed at the center of mass?
    world_com: Vector,
    solver_vel: u32,
}

struct JointConstraint {
    solver_vel_a: u32,
    solver_vel_b: u32,
    im_a: Vector,
    im_b: Vector,

    /// The constraints for a joint. Up to 6 in 3D, and up to 3 in 2D.
    elements: array<JointConstraintElement, Joint::SPATIAL_DIM>,
    /// The number of active `JointConstraint::elements`.
    len: u32,
}

struct JointConstraintElement {
    joint_id: u32,
    impulse: f32,
    impulse_bounds: vec2<f32>,
    lin_jac: Vector,
    ang_jac_a: AngVector,
    ang_jac_b: AngVector,
    ii_ang_jac_a: AngVector,
    ii_ang_jac_b: AngVector,
    inv_lhs: f32,
    rhs: f32,
    rhs_wo_bias: f32,
    cfm_gain: f32,
    cfm_coeff: f32,
}