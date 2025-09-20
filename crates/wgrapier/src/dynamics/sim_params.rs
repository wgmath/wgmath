use wgcore::Shader;
use wgparry::{dim_shader_defs, substitute_aliases};

#[derive(Shader)]
#[shader(src = "sim_params.wgsl")]
pub struct WgSimParams;

/// Parameters for a time-step of the physics engine.
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct GpuSimParams {
    /// The timestep length (default: `1.0 / 60.0`).
    pub dt: f32,
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
    pub min_ccd_dt: f32,

    /// > 0: the damping ratio used by the springs for contact constraint stabilization.
    ///
    /// Larger values make the constraints more compliant (allowing more visible
    /// penetrations before stabilization).
    /// (default `5.0`).
    pub contact_damping_ratio: f32,

    /// > 0: the natural frequency used by the springs for contact constraint regularization.
    ///
    /// Increasing this value will make it so that penetrations get fixed more quickly at the
    /// expense of potential jitter effects due to overshooting. In order to make the simulation
    /// look stiffer, it is recommended to increase the [`Self::contact_damping_ratio`] instead of this
    /// value.
    /// (default: `30.0`).
    pub contact_natural_frequency: f32,

    /// > 0: the natural frequency used by the springs for joint constraint regularization.
    ///
    /// Increasing this value will make it so that penetrations get fixed more quickly.
    /// (default: `1.0e6`).
    pub joint_natural_frequency: f32,

    /// The fraction of critical damping applied to the joint for constraints regularization.
    ///
    /// Larger values make the constraints more compliant (allowing more joint
    /// drift before stabilization).
    /// (default `1.0`).
    pub joint_damping_ratio: f32,

    /// The coefficient in `[0, 1]` applied to warmstart impulses, i.e., impulses that are used as the
    /// initial solution (instead of 0) at the next simulation step.
    ///
    /// This should generally be set to 1.
    ///
    /// (default `1.0`).
    pub warmstart_coefficient: f32,

    /// The approximate size of most dynamic objects in the scene.
    ///
    /// This value is used internally to estimate some length-based tolerance. In particular, the
    /// values [`GpuSimParams::allowed_linear_error`],
    /// [`GpuSimParams::max_corrective_velocity`],
    /// [`GpuSimParams::prediction_distance`], [`RigidBodyActivation::normalized_linear_threshold`]
    /// are scaled by this value implicitly.
    ///
    /// This value can be understood as the number of units-per-meter in your physical world compared
    /// to a human-sized world in meter. For example, in a 2d game, if your typical object size is 100
    /// pixels, set the [`Self::length_unit`] parameter to 100.0. The physics engine will interpret
    /// it as if 100 pixels is equivalent to 1 meter in its various internal threshold.
    /// (default `1.0`).
    pub length_unit: f32,

    /// Amount of penetration the engine won’t attempt to correct (default: `0.001m`).
    ///
    /// This value is implicitly scaled by [`GpuSimParams::length_unit`].
    pub normalized_allowed_linear_error: f32,
    /// Maximum amount of penetration the solver will attempt to resolve in one timestep (default: `10.0`).
    ///
    /// This value is implicitly scaled by [`GpuSimParams::length_unit`].
    pub normalized_max_corrective_velocity: f32,
    /// The maximal distance separating two objects that will generate predictive contacts (default: `0.002m`).
    ///
    /// This value is implicitly scaled by [`GpuSimParams::length_unit`].
    pub normalized_prediction_distance: f32,
    /// The number of solver iterations run by the constraints solver for calculating forces (default: `4`).
    pub num_solver_iterations: u32,
    /// Number of addition friction resolution iteration run during the last solver sub-step (default: `0`).
    pub num_additional_friction_iterations: u32,
    /// Number of internal Project Gauss Seidel (PGS) iterations run at each solver iteration (default: `1`).
    pub num_internal_pgs_iterations: u32,
    /// The number of stabilization iterations run at each solver iterations (default: `2`).
    pub num_internal_stabilization_iterations: u32,
    /// Minimum number of dynamic bodies in each active island (default: `128`).
    pub min_island_size: u32,
    /// Maximum number of substeps performed by the  solver (default: `1`).
    pub max_ccd_substeps: u32,
}

impl GpuSimParams {
    /// The inverse of the time-stepping length, i.e. the steps per seconds (Hz).
    ///
    /// This is zero if `self.dt` is zero.
    #[inline(always)]
    pub fn inv_dt(&self) -> f32 {
        if self.dt == 0.0 {
            0.0
        } else {
            1.0 / self.dt
        }
    }

    /// Sets the inverse time-stepping length (i.e. the frequency).
    ///
    /// This automatically recompute `self.dt`.
    #[inline]
    pub fn set_inv_dt(&mut self, inv_dt: f32) {
        if inv_dt == 0.0 {
            self.dt = 0.0
        } else {
            self.dt = 1.0 / inv_dt
        }
    }

    /// Initialize the simulation parameters with settings matching the TGS-soft solver
    /// with warmstarting.
    ///
    /// This is the default configuration, equivalent to [`GpuSimParams::default()`].
    pub fn tgs_soft() -> Self {
        Self {
            dt: 1.0 / 60.0,
            min_ccd_dt: 1.0 / 60.0 / 100.0,
            contact_natural_frequency: 30.0,
            contact_damping_ratio: 5.0,
            joint_natural_frequency: 1.0e6,
            joint_damping_ratio: 1.0,
            warmstart_coefficient: 1.0,
            num_internal_pgs_iterations: 1,
            num_internal_stabilization_iterations: 2,
            num_additional_friction_iterations: 0,
            num_solver_iterations: 4,
            // TODO: what is the optimal value for min_island_size?
            // It should not be too big so that we don't end up with
            // huge islands that don't fit in cache.
            // However we don't want it to be too small and end up with
            // tons of islands, reducing SIMD parallelism opportunities.
            min_island_size: 128,
            normalized_allowed_linear_error: 0.001,
            normalized_max_corrective_velocity: 10.0,
            normalized_prediction_distance: 0.002,
            max_ccd_substeps: 1,
            length_unit: 1.0,
        }
    }

    /// Initialize the simulation parameters with settings matching the TGS-soft solver
    /// **without** warmstarting.
    ///
    /// The [`GpuSimParams::tgs_soft()`] configuration should be preferred unless
    /// warmstarting proves to be undesirable for your use-case.
    pub fn tgs_soft_without_warmstart() -> Self {
        Self {
            contact_damping_ratio: 0.25,
            warmstart_coefficient: 0.0,
            num_additional_friction_iterations: 4,
            ..Self::tgs_soft()
        }
    }

    /// Initializes the integration parameters to match the legacy PGS solver from Rapier version <= 0.17.
    ///
    /// This exists mainly for testing and comparison purpose.
    pub fn pgs_legacy() -> Self {
        Self {
            num_solver_iterations: 1,
            num_internal_pgs_iterations: 4,
            ..Self::tgs_soft_without_warmstart()
        }
    }
}

impl Default for GpuSimParams {
    fn default() -> Self {
        Self::tgs_soft()
    }
}
