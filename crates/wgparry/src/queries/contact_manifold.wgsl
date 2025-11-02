//! Contact Manifold Module
//!
//! This module defines structures for storing contact information between colliders.
//! Contact manifolds are the output of narrow-phase collision detection and serve
//! as input to physics solvers.
//!
//! Dimension-specific limits:
//! - 2D: Up to 2 contact points per manifold
//! - 3D: Up to 4 contact points per manifold
//!
//! Contact points are stored with their location (in the first object's local frame)
//! and penetration distance. The manifold also includes a shared contact normal.

#define_import_path wgparry::contact_manifold


#if DIM == 2
/// Maximum number of contact points in a 2D contact manifold.
const MAX_MANIFOLD_POINTS: u32 = 2;
#else
/// Maximum number of contact points in a 3D contact manifold.
const MAX_MANIFOLD_POINTS: u32 = 4;
#endif

/// A single contact point within a manifold.
struct ContactPoint {
    // NOTE: field order is important here to make this struct as compact as possible.
    /// Contact point location (in object A's local frame).
    pt: Vector,
    /// Signed penetration distance (negative = penetrating).
    dist: f32,
}

/// A contact manifold containing multiple contact points.
///
/// Represents the contact region between two colliders. Multiple points
/// provide stability for physics simulation.
struct ContactManifold {
    // NOTE: fields order is important here to make this struct as compact as possible.
    points_a: array<ContactPoint, MAX_MANIFOLD_POINTS>,
    normal_a: Vector,
    len: u32,
}

fn single_point(pt: Vector, dist: f32, normal: Vector) -> ContactManifold {
    var result = ContactManifold();
    result.points_a[0] = ContactPoint(pt, dist);
    result.normal_a = normal;
    result.len = 1;
    return result;
}