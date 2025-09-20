#define_import_path wgrapier::dynamics::sim_params

const MAX_FLT: f32 = 3.40282347E+38;
const TWO_PI: f32 = 6.28318530717958647692528676655900577;

/// Parameters for a time-step of the physics engine.
struct SimParams {
    /// The timestep length (default: `1.0 / 60.0`).
    dt: f32,
    /// Minimum timestep size when using CCD with multiple substeps (default: `1.0 / 60.0 / 100.0`).
    ///
    /// When CCD with multiple substeps is enabled, the timestep is subdivided
    /// into smaller pieces. This timestep subdivision won't generate timestep
    /// lengths smaller than `min_ccd_dt`.
    ///
    /// Setting this to a large value will reduce the opportunity to performing
    /// CCD substepping, resulting in potentially more time dropped by the
    /// motion-clamping mechanism. Setting this to an very small value may lead
    /// to numerical instabilities.
    min_ccd_dt: f32,

    /// > 0: the damping ratio used by the springs for contact constraint stabilization.
    ///
    /// Larger values make the constraints more compliant (allowing more visible
    /// penetrations before stabilization).
    /// (default `5.0`).
    contact_damping_ratio: f32,

    /// > 0: the natural frequency used by the springs for contact constraint regularization.
    ///
    /// Increasing this value will make it so that penetrations get fixed more quickly at the
    /// expense of potential jitter effects due to overshooting. In order to make the simulation
    /// look stiffer, it is recommended to increase the [`Self::contact_damping_ratio`] instead of this
    /// value.
    /// (default: `30.0`).
    contact_natural_frequency: f32,

    /// > 0: the natural frequency used by the springs for joint constraint regularization.
    ///
    /// Increasing this value will make it so that penetrations get fixed more quickly.
    /// (default: `1.0e6`).
    joint_natural_frequency: f32,

    /// The fraction of critical damping applied to the joint for constraints regularization.
    ///
    /// Larger values make the constraints more compliant (allowing more joint
    /// drift before stabilization).
    /// (default `1.0`).
    joint_damping_ratio: f32,

    /// The coefficient in `[0, 1]` applied to warmstart impulses, i.e., impulses that are used as the
    /// initial solution (instead of 0) at the next simulation step.
    ///
    /// This should generally be set to 1.
    ///
    /// (default `1.0`).
    warmstart_coefficient: f32,

    /// The approximate size of most dynamic objects in the scene.
    ///
    /// This value is used internally to estimate some length-based tolerance. In particular, the
    /// values [`IntegrationParameters::allowed_linear_error`],
    /// [`IntegrationParameters::max_corrective_velocity`],
    /// [`IntegrationParameters::prediction_distance`], [`RigidBodyActivation::normalized_linear_threshold`]
    /// are scaled by this value implicitly.
    ///
    /// This value can be understood as the number of units-per-meter in your physical world compared
    /// to a human-sized world in meter. For example, in a 2d game, if your typical object size is 100
    /// pixels, set the [`Self::length_unit`] parameter to 100.0. The physics engine will interpret
    /// it as if 100 pixels is equivalent to 1 meter in its various internal threshold.
    /// (default `1.0`).
    length_unit: f32,

    /// Amount of penetration the engine won’t attempt to correct (default: `0.001m`).
    ///
    /// This value is implicitly scaled by [`IntegrationParameters::length_unit`].
    normalized_allowed_linear_error: f32,
    /// Maximum amount of penetration the solver will attempt to resolve in one timestep (default: `10.0`).
    ///
    /// This value is implicitly scaled by [`IntegrationParameters::length_unit`].
    normalized_max_corrective_velocity: f32,
    /// The maximal distance separating two objects that will generate predictive contacts (default: `0.002m`).
    ///
    /// This value is implicitly scaled by [`IntegrationParameters::length_unit`].
    normalized_prediction_distance: f32,
    /// The number of solver iterations run by the constraints solver for calculating forces (default: `4`).
    num_solver_iterations: u32,
    /// Number of addition friction resolution iteration run during the last solver sub-step (default: `0`).
    num_additional_friction_iterations: u32,
    /// Number of internal Project Gauss Seidel (PGS) iterations run at each solver iteration (default: `1`).
    num_internal_pgs_iterations: u32,
    /// The number of stabilization iterations run at each solver iterations (default: `2`).
    num_internal_stabilization_iterations: u32,
    /// Minimum number of dynamic bodies in each active island (default: `128`).
    min_island_size: u32,
    /// Maximum number of substeps performed by the  solver (default: `1`).
    max_ccd_substeps: u32,
}

/// The inverse of the time-stepping length, i.e. the steps per seconds (Hz).
///
/// This is zero if `params.dt` is zero.
fn inv_dt(params: SimParams) -> f32 {
    if params.dt == 0.0 {
        return 0.0;
    } else {
        return 1.0 / params.dt;
    }
}

/// The contact’s spring angular frequency for constraints regularization.
fn contact_angular_frequency(params: SimParams) -> f32 {
    return params.contact_natural_frequency * TWO_PI;
}

/// The [`Self::contact_erp`] coefficient, multiplied by the inverse timestep length.
fn contact_erp_inv_dt(params: SimParams) -> f32 {
    let ang_freq = contact_angular_frequency(params);
    return ang_freq / (params.dt * ang_freq + 2.0 * params.contact_damping_ratio);
}

/// The effective Error Reduction Parameter applied for calculating regularization forces
/// on contacts.
///
/// This parameter is computed automatically from [`Self::contact_natural_frequency`],
/// [`Self::contact_damping_ratio`] and the substep length.
fn contact_erp(params: SimParams) -> f32 {
    return params.dt * contact_erp_inv_dt(params);
}

/// The joint’s spring angular frequency for constraint regularization.
fn joint_angular_frequency(params: SimParams) -> f32 {
    return params.joint_natural_frequency * TWO_PI;
}

/// The [`Self::joint_erp`] coefficient, multiplied by the inverse timestep length.
fn joint_erp_inv_dt(params: SimParams) -> f32 {
    let ang_freq = joint_angular_frequency(params);
    return ang_freq / (params.dt * ang_freq + 2.0 * params.joint_damping_ratio);
}

/// The effective Error Reduction Parameter applied for calculating regularization forces
/// on joints.
///
/// This parameter is computed automatically from [`Self::joint_natural_frequency`],
/// [`Self::joint_damping_ratio`] and the substep length.
fn joint_erp(params: SimParams) -> f32 {
    return params.dt * joint_erp_inv_dt(params);
}

/// The CFM factor to be used in the constraint resolution.
///
/// This parameter is computed automatically from [`Self::contact_natural_frequency`],
/// [`Self::contact_damping_ratio`] and the substep length.
fn contact_cfm_factor(params: SimParams) -> f32 {
    // Compute CFM assuming a critically damped spring multiplied by the damping ratio.
    // The logic is similar to [`Self::joint_cfm_coeff`].
    let contact_erp = contact_erp(params);
    if contact_erp == 0.0 {
        return 0.0;
    }
    let inv_erp_minus_one = 1.0 / contact_erp - 1.0;

    // let stiffness = 4.0 * damping_ratio * damping_ratio * projected_mass
    //     / (dt * dt * inv_erp_minus_one * inv_erp_minus_one);
    // let damping = 4.0 * damping_ratio * damping_ratio * projected_mass
    //     / (dt * inv_erp_minus_one);
    // let cfm = 1.0 / (dt * dt * stiffness + dt * damping);
    // NOTE: This simplifies to cfm = cfm_coeff / projected_mass:
    let cfm_coeff = inv_erp_minus_one * inv_erp_minus_one
        / ((1.0 + inv_erp_minus_one)
            * 4.0
            * params.contact_damping_ratio
            * params.contact_damping_ratio);

    // Furthermore, we use this coefficient inside of the impulse resolution.
    // Surprisingly, several simplifications happen there.
    // Let `m` the projected mass of the constraint.
    // Let `m’` the projected mass that includes CFM: `m’ = 1 / (1 / m + cfm_coeff / m) = m / (1 + cfm_coeff)`
    // We have:
    // new_impulse = old_impulse - m’ (delta_vel - cfm * old_impulse)
    //             = old_impulse - m / (1 + cfm_coeff) * (delta_vel - cfm_coeff / m * old_impulse)
    //             = old_impulse * (1 - cfm_coeff / (1 + cfm_coeff)) - m / (1 + cfm_coeff) * delta_vel
    //             = old_impulse / (1 + cfm_coeff) - m * delta_vel / (1 + cfm_coeff)
    //             = 1 / (1 + cfm_coeff) * (old_impulse - m * delta_vel)
    // So, setting cfm_factor = 1 / (1 + cfm_coeff).
    // We obtain:
    // new_impulse = cfm_factor * (old_impulse - m * delta_vel)
    //
    // The value returned by this function is this cfm_factor that can be used directly
    // in the constraint solver.
    return 1.0 / (1.0 + cfm_coeff);
}

/// The CFM (constraints force mixing) coefficient applied to all joints for constraints regularization.
///
/// This parameter is computed automatically from [`Self::joint_natural_frequency`],
/// [`Self::joint_damping_ratio`] and the substep length.
fn joint_cfm_coeff(params: SimParams) -> f32 {
    // Compute CFM assuming a critically damped spring multiplied by the damping ratio.
    // The logic is similar to `Self::contact_cfm_factor`.
    let joint_erp = joint_erp(params);
    if joint_erp == 0.0 {
        return 0.0;
    }
    let inv_erp_minus_one = 1.0 / joint_erp - 1.0;
    return inv_erp_minus_one * inv_erp_minus_one
        / ((1.0 + inv_erp_minus_one)
            * 4.0
            * params.joint_damping_ratio
            * params.joint_damping_ratio);
}

/// Amount of penetration the engine won’t attempt to correct (default: `0.001` multiplied by
/// [`Self::length_unit`]).
fn allowed_linear_error(params: SimParams) -> f32 {
    return params.normalized_allowed_linear_error * params.length_unit;
}

/// Maximum amount of penetration the solver will attempt to resolve in one timestep.
///
/// This is equal to [`Self::normalized_max_corrective_velocity`] multiplied by
/// [`Self::length_unit`].
fn max_corrective_velocity(params: SimParams) -> f32 {
    if params.normalized_max_corrective_velocity != MAX_FLT {
        return params.normalized_max_corrective_velocity * params.length_unit;
    } else {
        return MAX_FLT;
    }
}

/// The maximal distance separating two objects that will generate predictive contacts
/// (default: `0.002m` multiped by [`Self::length_unit`]).
fn prediction_distance(params: SimParams) -> f32 {
    return params.normalized_prediction_distance * params.length_unit;
}
