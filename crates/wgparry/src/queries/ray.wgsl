//! Ray Definition and Operations
//!
//! This module provides the ray structure and basic ray operations.
//!
//! Note: Currently only supports 3D rays. 2D ray support is not yet implemented.
//!
//! A ray is defined by an origin point and a direction vector.
//! The ray represents all points: origin + t * dir for t >= 0.

#define_import_path wgparry::ray

/// A ray in 3D space.
///
/// Represents a half-line starting at 'origin' and extending infinitely
/// in the 'dir' direction.
///
/// Note: The direction vector does not need to be normalized.
struct Ray {
    /// The ray's starting point.
    origin: vec3<f32>,
    /// The ray's direction vector.
    dir: vec3<f32>,
}

/// Computes a point on the ray at parameter t.
///
/// Parameters:
/// - ray: The ray.
/// - t: The parameter (t >= 0 for points on the ray).
///
/// Returns: The point at `origin + t * dir`.
fn ptAt(ray: Ray, t: f32) -> vec3<f32> {
    return ray.origin + ray.dir * t;
}