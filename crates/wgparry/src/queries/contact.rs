//! Contact generation for collision response.
//!
//! This module implements contact manifold generation between pairs of colliding shapes.
//! Contact manifolds contain multiple contact points with normals and penetration depths,
//! which are essential for physics simulation and collision response.
//!
//! # Contact Manifolds
//!
//! A contact manifold represents a collision between two shapes and contains:
//! - **Contact points**: Up to 2 points in 2D, 4 points in 3D.
//! - **Contact normal**: The direction to separate the shapes.
//! - **Penetration depths**: How deep each contact point penetrates.

use super::{WgPolygonalFeature, WgSat};
use crate::math::Vector;
use crate::queries::gjk::{WgCsoPoint, WgEpa, WgGjk, WgVoronoiSimplex};
use crate::shapes::{WgBall, WgCuboid, WgShape};
use crate::{dim_shader_defs, substitute_aliases};
use encase::ShaderType;
use na::Vector2;
use wgcore::Shader;
use wgebra::{WgSim2, WgSim3};

#[cfg(feature = "dim3")]
use crate::shapes::{WgCone, WgCylinder};

#[cfg(feature = "dim2")]
const MAX_MANIFOLD_POINTS: usize = 2;
#[cfg(feature = "dim3")]
const MAX_MANIFOLD_POINTS: usize = 4;

/// A single contact point within a contact manifold.
///
/// Each contact point represents a location where two shapes are touching or penetrating,
/// with an associated penetration distance (negative for separation, positive for overlap).
#[derive(Copy, Clone, PartialEq, Debug, ShaderType)]
pub struct GpuContactPoint {
    /// The contact point position in world space.
    point: Vector<f32>,

    /// Signed distance (penetration depth).
    ///
    /// - Positive values indicate penetration (shapes overlap)
    /// - Negative values indicate separation (shapes don't touch)
    /// - Zero indicates contact at the surface
    dist: f32,
}

/// A contact manifold containing multiple contact points.
///
/// The manifold describes a collision between two shapes with:
/// - Up to 2 contact points in 2D or 4 contact points in 3D.
/// - A shared contact normal.
/// - The actual number of valid contact points.
///
/// # Stability
///
/// Multiple contact points provide rotational stability in physics simulation.
/// For example, a box resting on a plane will have 2-4 contact points depending
/// on orientation, preventing unrealistic tipping.
#[derive(Copy, Clone, PartialEq, Debug, ShaderType)]
pub struct GpuContactManifold {
    /// Array of contact points.
    ///
    /// Only the first `len` points are valid; remaining slots may contain garbage data.
    points: [GpuContactPoint; MAX_MANIFOLD_POINTS],

    /// The contact normal pointing from shape A toward shape B.
    ///
    /// This is the direction along which the shapes should be separated to resolve
    /// the collision.
    normal: Vector<f32>,

    /// Number of valid contact points in the `points` array.
    ///
    /// Valid range: 0 to [`MAX_MANIFOLD_POINTS`]
    len: u32,
}

/// An indexed contact associating a manifold with two collider indices.
///
/// Used in broad-phase collision detection to store contacts along with the
/// identifiers of the colliding objects.
#[derive(Copy, Clone, PartialEq, Debug, ShaderType)]
pub struct GpuIndexedContact {
    /// The contact manifold describing the collision.
    contact: GpuContactManifold,

    /// Indices of the two colliding objects `[collider_a, collider_b]`.
    colliders: Vector2<u32>,
}

#[derive(Shader)]
#[cfg_attr(
    feature = "dim2",
    shader(
        derive(
            WgBall,
            WgCuboid,
            WgShape,
            WgSim2,
            WgSim3,
            WgSat,
            WgPolygonalFeature,
            WgContactManifold,
            WgContactPfmPfm,
        ),
        src = "contact.wgsl",
        src_fn = "substitute_aliases",
        shader_defs = "dim_shader_defs"
    )
)]
#[cfg_attr(
    feature = "dim3",
    shader(
        derive(
            WgBall,
            WgCuboid,
            WgShape,
            WgSim2,
            WgSim3,
            WgSat,
            WgPolygonalFeature,
            WgContactManifold,
            WgContactPfmPfm,
            WgCylinder,
            WgCone
        ),
        src = "contact.wgsl",
        src_fn = "substitute_aliases",
        shader_defs = "dim_shader_defs"
    )
)]
/// GPU shader for contact generation between shapes.
///
/// This shader implements collision detection algorithms that generate contact
/// manifolds for various shape type pairs. It composes specialized algorithms:
/// - Ball-ball contacts (simple distance checks)
/// - Cuboid-cuboid contacts (SAT + polygonal feature clipping)
/// - Mixed shape contacts (ball-cuboid, etc.)
pub struct WgContact;

#[derive(Shader)]
#[shader(
    derive(WgSim2, WgSim3),
    src = "contact_manifold.wgsl",
    src_fn = "substitute_aliases",
    shader_defs = "dim_shader_defs"
)]
/// GPU shader defining contact manifold data structures.
///
/// Provides the WGSL definitions for [`GpuContactPoint`], [`GpuContactManifold`],
/// and [`GpuIndexedContact`], along with utility functions for manipulating
/// contact manifolds.
///
/// This shader is a dependency for [`WgContact`] and other contact-related shaders.
pub struct WgContactManifold;

#[derive(Shader)]
#[shader(
    derive(
        WgCsoPoint,
        WgVoronoiSimplex,
        WgShape,
        WgGjk,
        WgEpa,
        WgPolygonalFeature,
    ),
    src = "contact_pfm_pfm.wgsl",
    src_fn = "substitute_aliases",
    shader_defs = "dim_shader_defs"
)]
/// Kernel for contact manifold calculation between two "Polygonal Feature Maps".
///
/// A Polygonal Feature Map is similar to the concept of Support Mappings, except that
/// instead associating an extremal point to a vector, it associates an extremal polygonal
/// face to the direction.
pub struct WgContactPfmPfm;

wgcore::test_shader_compilation!(WgContact, wgcore, crate::dim_shader_defs());
wgcore::test_shader_compilation!(WgContactPfmPfm, wgcore, crate::dim_shader_defs());
