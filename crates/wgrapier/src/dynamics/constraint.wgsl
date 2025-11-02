//! Contact constraint data structures
//!
//! This shader defines the data structures for contact constraints used in the
//! iterative constraint solver.
//!
//! Each contact manifold (a set of contact points between two bodies) generates:
//! - Normal constraints: Prevent penetration along contact normal.
//! - Tangent constraints: Model friction in tangent directions.
//!
//! The constraint solver iteratively applies impulses to satisfy:
//! 1. Non-penetration: C_n >= 0 (unilateral constraint)
//! 2. Friction: |C_t| <= μ * C_n (Coulomb friction cone)
//!
//! Solver Coefficients:
//! - r: Inverse effective mass (1 / (m_eff))
//! - rhs: Right-hand side (target relative velocity)
//! - impulse: Accumulated impulse (Lagrange multiplier)
//!
//! Dimension-Specific Details:
//! - 2D: SUB_LEN=1 (one tangent), MAX_CONSTRAINTS=2 contact points
//! - 3D: SUB_LEN=2 (two tangents), MAX_CONSTRAINTS=4 contact points

#define_import_path wgrapier::dynamics::constraint


#if DIM == 2
/// Number of tangent constraint directions (2D: one tangent perpendicular to normal).
const SUB_LEN: u32 = 1;

/// Maximum number of contact points per contact manifold (2D: typically 2).
const MAX_CONSTRAINTS_PER_MANIFOLD: u32 = 2;
#else
/// Number of tangent constraint directions (3D: two tangents in contact plane).
const SUB_LEN: u32 = 2;

/// Maximum number of contact points per contact manifold (3D: up to 4).
const MAX_CONSTRAINTS_PER_MANIFOLD: u32 = 4;
#endif

/// Metadata for building a two-body constraint from a contact point.
///
/// This data is used to initialize and update constraints at each solver substep.
struct TwoBodyConstraintInfos {
    /// Tangent velocity (for conveyor belt effects, currently unused).
    tangent_vel: Vector, // TODO PERF: could be one float less, be shared by both contact point infos?

    /// Normal relative velocity at the contact point (for restitution).
    normal_vel: f32,

    /// Contact point in body A's local coordinates (for warmstarting).
    /// Stored in local space to detect matching contacts across frames.
    local_pt_a: Vector,

    /// Contact point in body B's local coordinates (for warmstarting).
    local_pt_b: Vector,

    /// Penetration depth (negative = penetration, positive = separation).
    dist: f32,
}

/// Builder data for constructing constraints from contact manifolds.
///
/// Stores auxiliary information needed to update constraints at each solver substep.
struct TwoBodyConstraintBuilder {
    /// Information for each contact point in the manifold.
    infos: array<TwoBodyConstraintInfos, MAX_CONSTRAINTS_PER_MANIFOLD>,
}

/// A contact constraint between two rigid bodies.
///
/// Encodes all the data needed to solve a contact constraint, including:
/// - Constraint directions (normal and tangent).
/// - Effective masses and inverse masses.
/// - Solver coefficients (CFM factor, friction limit).
/// - Per-contact-point constraint elements.
// PERF: differentiate two-bodies and one-body constraints?
struct TwoBodyConstraint {
    /// Contact normal direction from body A's perspective (points away from A).
    /// Normal impulses are applied along this direction to prevent penetration.
    dir_a: Vector, // Non-penetration force direction for the first body.

#if DIM == 3
    /// First tangent direction (3D only, orthogonal to normal).
    /// Used for friction in the contact plane.
    tangent_a: Vector, // One of the friction force directions.
#endif

    /// Inverse mass of body A along each axis.
    /// Used to compute linear velocity changes from impulses.
    im_a: Vector,

    /// Inverse mass of body B along each axis.
    im_b: Vector,

    /// Constraint Force Mixing (CFM) factor for regularization.
    /// Softens the constraint: new_impulse = cfm_factor * (old_impulse - ...)
    cfm_factor: f32,

    /// Friction coefficient (μ in Coulomb friction model).
    /// Friction impulse magnitude limited to: |f_tangent| <= limit * f_normal
    limit: f32,

    /// Index of body A in the solver arrays.
    solver_body_a: u32,

    /// Index of body B in the solver arrays.
    solver_body_b: u32,

    /// Per-contact-point constraint data (up to MAX_CONSTRAINTS_PER_MANIFOLD).
    elements: array<TwoBodyConstraintElement, MAX_CONSTRAINTS_PER_MANIFOLD>,

    /// Number of active contact points in this manifold (1-4 in 3D, 1-2 in 2D).
    len: u32,
}

/// Constraint data for a single contact point.
struct TwoBodyConstraintElement {
    /// Normal constraint: prevents penetration.
    normal_part: TwoBodyConstraintNormalPart,

    /// Tangent constraint(s): models friction.
    tangent_part: TwoBodyConstraintTangentPart,
}

/// Normal constraint data (non-penetration).
///
/// Implements the constraint: C >= 0 (bodies cannot interpenetrate)
/// Solved as a unilateral constraint (impulse >= 0).
struct TwoBodyConstraintNormalPart {
    /// Angular contribution for body A: r_a × normal
    /// (In 2D: scalar cross product, in 3D: vector cross product)
    gcross_a: AngVector,

    /// Angular contribution for body B: r_b × normal
    gcross_b: AngVector,

    /// Right-hand side: target relative velocity (includes bias for correction).
    /// rhs = desired_velocity + bias_velocity
    rhs: f32,

    /// Right-hand side without bias term (used in substep iterations).
    /// rhs_wo_bias = desired_velocity (without position correction)
    rhs_wo_bias: f32,

    /// Current impulse magnitude for this iteration.
    /// Updated during solving, used for warmstarting next frame.
    impulse: f32,

    /// Second impulse values specific to the jacobi solver.
    impulse_jacobi: f32,

    /// Inverse effective mass: 1 / (m_eff)
    /// where m_eff = projected mass along constraint direction
    r: f32,
}

/// Tangent constraint data (friction).
///
/// Implements Coulomb friction: |f_tangent| <= μ * f_normal
/// Solved as a bilateral constraint with limits.
struct TwoBodyConstraintTangentPart {
    /// Angular contributions for body A (one per tangent direction).
    gcross_a: array<AngVector, SUB_LEN>,

    /// Angular contributions for body B (one per tangent direction).
    gcross_b: array<AngVector, SUB_LEN>,

    /// Right-hand sides (one per tangent direction).
    rhs: array<f32, SUB_LEN>,

    /// Right-hand sides without bias (one per tangent direction).
    rhs_wo_bias: array<f32, SUB_LEN>,

#if DIM == 2
    /// Current tangent impulse (2D: single scalar for one tangent direction).
    impulse: array<f32, 1>,

    /// Second impulse values specific to the jacobi solver.
    impulse_jacobi: array<f32, 1>,

    /// Inverse effective mass (2D: single value).
    r: array<f32, 1>,
#else
    /// Current tangent impulses (3D: vec2 for two tangent directions).
    impulse: vec2<f32>,

    /// Second impulse values specific to the jacobi solver.
    impulse_jacobi: vec2<f32>,

    /// Inverse effective mass components (3D: 3 values for 2x2 mass matrix).
    /// r[0] = r_00, r[1] = r_11, r[2] = r_01 (symmetric, so r_10 = r_01)
    r: array<f32, 3>,
#endif
}