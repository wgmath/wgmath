#define_import_path wgrapier::dynamics::joint

#if DIM == 2
    #import wgebra::sim2 as Pose
#else
    #import wgebra::sim3 as Pose
#endif

#if DIM == 2
const SPATIAL_DIM: u32 = 3;
#else
const SPATIAL_DIM: u32 = 6;
#endif

#if DIM == 2
const LIN_AXES_MASK: u32 = 1 + (1 << 1) + (1 << 2);
const ANG_AXES_MASK: u32 = (1 << 3) + (1 << 4) + (1 << 5);
#else
const LIN_AXES_MASK: u32 = 1 + (1 << 1);
const ANG_AXES_MASK: u32 = 1 << 2;
#endif

struct ImpulseJoint {
    body_a: u32,
    body_b: u32,
    data: GenericJoint,
}

/// A generic (6 DOFs in 3D or 3 DOFs in 2D) joint.
struct GenericJoint {
    /// The joint’s frame, expressed in the first rigid-body’s local-space.
    local_frame_a: Transform,
    /// The joint’s frame, expressed in the second rigid-body’s local-space.
    local_frame_b: Transform,
    /// The degrees-of-freedoms locked by this joint.
    locked_axes: u32,
    /// The degrees-of-freedoms limited by this joint.
    limit_axes: u32,
    /// The degrees-of-freedoms motorised by this joint.
    motor_axes: u32,
    /// The coupled degrees of freedom of this joint.
    ///
    /// Note that coupling degrees of freedoms (DoF) changes the interpretation of the coupled joint’s limits and motors.
    /// If multiple linear DoF are limited/motorized, only the limits/motor configuration for the first
    /// coupled linear DoF is applied to all coupled linear DoF. Similarly, if multiple angular DoF are limited/motorized
    /// only the limits/motor configuration for the first coupled angular DoF is applied to all coupled angular DoF.
    coupled_axes: u32,
    /// The limits, along each degree of freedoms of this joint.
    ///
    /// Note that the limit must also be explicitly enabled by the `limit_axes` bitmask.
    /// For coupled degrees of freedoms (DoF), only the first linear (resp. angular) coupled DoF limit and `limit_axis`
    /// bitmask is applied to the coupled linear (resp. angular) axes.
    limits: array<JointLimits, SPATIAL_DIM>,
    /// The motors, along each degree of freedoms of this joint.
    ///
    /// Note that the motor must also be explicitly enabled by the `motor_axes` bitmask.
    /// For coupled degrees of freedoms (DoF), only the first linear (resp. angular) coupled DoF motor and `motor_axes`
    /// bitmask is applied to the coupled linear (resp. angular) axes.
    motors: array<JointMotor, SPATIAL_DIM>,
}

/// Limits that restrict a joint's range of motion along one axis.
///
/// Use to constrain how far a joint can move/rotate. Examples:
/// - Door that only opens 90°: revolute joint with limits `[0.0, PI/2.0]`
/// - Piston with 2-unit stroke: prismatic joint with limits `[0.0, 2.0]`
/// - Elbow that bends 0-150°: revolute joint with limits `[0.0, 5*PI/6]`
///
/// When a joint hits its limit, forces are applied to prevent further movement in that direction.
struct JointLimits {
    /// Minimum allowed value (angle for revolute, distance for prismatic).
    min: f32,
    /// Maximum allowed value (angle for revolute, distance for prismatic).
    max: f32,
    /// Internal: impulse being applied to enforce the limit.
    impulse: f32,
}

/// A powered motor that drives a joint toward a target position/velocity.
///
/// Motors add actuation to joints - they apply forces to make the joint move toward
/// a desired state. Think of them as servos, electric motors, or hydraulic actuators.
///
/// ## Two control modes
///
/// 1. **Velocity control**: Set `target_vel` to make the motor spin/slide at constant speed
/// 2. **Position control**: Set `target_pos` with `stiffness`/`damping` to reach a target angle/position
struct JointMotor {
    /// Target velocity (units/sec for prismatic, rad/sec for revolute).
    target_vel: f32,
    /// Target position (units for prismatic, radians for revolute).
    target_pos: f32,
    /// Spring constant - how strongly to pull toward target position.
    stiffness: f32,
    /// Damping coefficient - resistance to motion (prevents oscillation).
    damping: f32,
    /// Maximum force the motor can apply (Newtons for prismatic, Nm for revolute).
    max_force: f32,
    /// Internal: current impulse being applied.
    impulse: f32,
    /// Force-based or acceleration-based motor model.
    model: u32,
}

/// Spring constants auto-scale with mass (easier to tune, recommended).
const ACCELERATION_BASED: u32 = 0;
/// Spring constants produce absolute forces (mass-dependent).
const FORCE_BASED: u32 = 1;

struct MotorParameters {
    erp_inv_dt: f32,
    cfm_coeff: f32,
    cfm_gain: f32,
    target_pos: f32,
    target_vel: f32,
    max_impulse: f32,
}

fn motor_params(motor: JointMotor, dt: f32) -> MotorParameters {
    if motor.model == ACCELERATION_BASED {
        let erp_inv_dt = motor.stiffness * pseudo_inv(dt * motor.stiffness + motor.damping);
        let cfm_coeff = pseudo_inv(dt * dt * motor.stiffness + dt * motor.damping);

        return MotorParameters(
            erp_inv_dt,
            cfm_coeff,
            0.0,
            motor.target_pos,
            motor.target_vel,
            motor.max_force * dt,
        );
    } else { // FORCE_BASED
        let erp_inv_dt = motor.stiffness * pseudo_inv(dt * motor.stiffness + motor.damping);
        let cfm_gain = pseudo_inv(dt * dt * motor.stiffness + dt * motor.damping);

        return MotorParameters(
            erp_inv_dt,
            0.0, // cfm_coeff,
            cfm_gain,
            motor.target_pos,
            motor.target_vel,
            motor.max_force * dt,
        );
    }
}

fn pseudo_inv(x: f32) -> f32 {
    return select(1.0 / x, 0.0, x == 0.0);
}