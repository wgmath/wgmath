//! Contact constraints for collision resolution.
//!
//! This module defines the constraint structures used to resolve collisions between rigid bodies.
//! Constraints are generated from contact manifolds and solved iteratively to compute impulses
//! that prevent penetration and simulate friction.

use encase::ShaderType;
use wgcore::Shader;
use wgparry::math::{AngVector, Vector};
use wgparry::{dim_shader_defs, queries::WgContact, substitute_aliases};

#[cfg(feature = "dim3")]
use nalgebra::Vector2;

/// WGSL shader defining constraint structures and helper functions.
///
/// This shader can be imported by constraint solver shaders to access constraint
/// data structures and computation helpers.
#[derive(Shader)]
#[shader(
    derive(WgContact),
    src = "constraint.wgsl",
    src_fn = "substitute_aliases",
    shader_defs = "dim_shader_defs"
)]
pub struct WgConstraint;

#[cfg(feature = "dim2")]
const SUB_LEN: usize = 1;
#[cfg(feature = "dim2")]
const MAX_CONSTRAINTS_PER_MANIFOLD: usize = 2;

#[cfg(feature = "dim3")]
const SUB_LEN: usize = 2;
#[cfg(feature = "dim3")]
const MAX_CONSTRAINTS_PER_MANIFOLD: usize = 4;

/// Geometric information for one contact point within a constraint.
///
/// This structure stores the contact geometry and relative velocities needed
/// to build the constraint's solver representation.
#[derive(Copy, Clone, PartialEq, Debug, ShaderType)]
pub struct GpuTwoBodyConstraintInfos {
    /// Relative tangential velocity at the contact point.
    pub tangent_vel: Vector<f32>,
    /// Relative normal velocity at the contact point.
    pub normal_vel: f32,
    /// Contact point in body A's local space.
    pub local_pt_a: Vector<f32>,
    /// Contact point in body B's local space.
    pub local_pt_b: Vector<f32>,
    /// Penetration distance (negative for separation).
    pub dist: f32,
}

/// Builder structure containing geometric info for all contact points in a manifold.
///
/// This is an intermediate representation used during constraint initialization.
/// It stores the raw contact geometry before computing the solver representation.
#[derive(Copy, Clone, PartialEq, Debug, ShaderType)]
pub struct GpuTwoBodyConstraintBuilder {
    /// Array of contact point information (up to 2 points in 2D, 4 in 3D).
    pub infos: [GpuTwoBodyConstraintInfos; MAX_CONSTRAINTS_PER_MANIFOLD],
}

/// A two-body contact constraint ready for iterative solving.
///
/// This structure contains all the precomputed data needed by the constraint solver,
/// including effective masses, Jacobian terms, impulse accumulators, and target
/// velocities (right-hand sides).
///
/// Each constraint can contain multiple contact points (up to 2 in 2D, 4 in 3D),
/// with both normal and tangent components for each point.
#[derive(Copy, Clone, PartialEq, Debug, ShaderType)]
pub struct GpuTwoBodyConstraint {
    /// Contact normal direction for body A.
    pub dir_a: Vector<f32>,
    /// First tangent direction for friction (3D only).
    #[cfg(feature = "dim3")]
    pub tangent_a: Vector<f32>,
    /// Inverse mass of body A.
    pub im_a: Vector<f32>,
    /// Inverse mass of body B.
    pub im_b: Vector<f32>,
    /// Constraint force mixing factor for regularization.
    pub cfm_factor: f32,
    /// Friction cone limit (max tangential impulse).
    pub limit: f32,
    /// Index of body A in the solver arrays.
    pub solver_body_a: u32,
    /// Index of body B in the solver arrays.
    pub solver_body_b: u32,
    /// Array of constraint elements, one per contact point.
    pub elements: [GpuTwoBodyConstraintElement; MAX_CONSTRAINTS_PER_MANIFOLD],
    /// Number of active contact points in this constraint.
    pub len: u32,
}

/// One element (contact point) within a two-body constraint.
///
/// Contains both the normal and tangent parts for a single contact point.
#[derive(Copy, Clone, PartialEq, Debug, ShaderType)]
pub struct GpuTwoBodyConstraintElement {
    /// Normal constraint component (prevents penetration).
    pub normal_part: GpuTwoBodyConstraintNormalPart,
    /// Tangent constraint component (applies friction).
    pub tangent_part: GpuTwoBodyConstraintTangentPart,
}

/// Normal (non-penetration) constraint data for one contact point.
///
/// This stores the Jacobian terms, effective mass, impulse accumulators,
/// and right-hand side for the normal constraint.
#[derive(Copy, Clone, PartialEq, Debug, ShaderType)]
pub struct GpuTwoBodyConstraintNormalPart {
    /// Angular Jacobian component for body A.
    pub torque_dir_a: AngVector<f32>,
    /// Angular Jacobian component for body A, multiplied by the inverse angular inertia.
    pub ii_torque_dir_a: AngVector<f32>,
    /// Angular Jacobian component for body B.
    pub torque_dir_b: AngVector<f32>,
    /// Angular Jacobian component for body B, multiplied by the inverse angular inertia.
    pub ii_torque_dir_b: AngVector<f32>,
    /// Right-hand side with bias term (used in stabilization phase).
    pub rhs: f32,
    /// Right-hand side without bias term (used in velocity solving phase).
    pub rhs_wo_bias: f32,
    /// Current iteration's impulse.
    pub impulse: f32,
    /// Accumulated impulse from all iterations (for clamping and warmstart).
    pub impulse_accumulator: f32,
    /// Effective mass (inverse of the constraint's mass matrix).
    pub r: f32,
}

/// Tangent (friction) constraint data for one contact point.
///
/// This stores the Jacobian terms, effective masses, impulse accumulators,
/// and right-hand sides for the friction constraints. In 2D there is one
/// tangent direction; in 3D there are two.
#[derive(Copy, Clone, PartialEq, Debug, ShaderType)]
pub struct GpuTwoBodyConstraintTangentPart {
    /// Angular Jacobian components for body A (one or two directions).
    pub torque_dir_a: [AngVector<f32>; SUB_LEN],
    /// Angular Jacobian components for body A (one or two directions) multiplied by the angular inertia tensor.
    pub ii_torque_dir_a: [AngVector<f32>; SUB_LEN],
    /// Angular Jacobian components for body B (one or two directions).
    pub torque_dir_b: [AngVector<f32>; SUB_LEN],
    /// Angular Jacobian components for body B (one or two directions) multiplied by the angular inertia tensor.
    pub ii_torque_dir_b: [AngVector<f32>; SUB_LEN],
    /// Right-hand sides with bias terms.
    pub rhs: [f32; SUB_LEN],
    /// Right-hand sides without bias terms.
    pub rhs_wo_bias: [f32; SUB_LEN],

    /// Current iteration's impulses (2D: 1 direction).
    #[cfg(feature = "dim2")]
    pub impulse: [f32; 1],
    /// Accumulated impulses (2D: 1 direction).
    #[cfg(feature = "dim2")]
    pub impulse_accumulator: [f32; 1],
    /// Effective masses (2D: 1 value).
    #[cfg(feature = "dim2")]
    pub r: [f32; 1],
    /// Current iteration's impulses (3D: 2 directions).
    #[cfg(feature = "dim3")]
    pub impulse: Vector2<f32>,
    /// Accumulated impulses (3D: 2 directions).
    #[cfg(feature = "dim3")]
    pub impulse_accumulator: Vector2<f32>,
    /// Effective masses (3D: 3 values for 2x2 symmetric matrix).
    #[cfg(feature = "dim3")]
    pub r: [f32; 3],
}
