//! Point Projection Result Module
//!
//! This module defines the common return type for all point projection operations.
//! `ProjectionResult` carries both the projected point location and information about
//! whether the original point was inside the shape.

#define_import_path wgparry::projection

/// The result of a point projection operation.
///
/// This structure is returned by all point projection functions.
struct ProjectionResult {
    /// The point’s projection on the shape.
    /// This can be equal to the original point if the point was inside
    /// of the shape and the projection function doesn’t always project
    /// on the boundary.
    point: Vector,
    /// Is the point inside of the shape?
    is_inside: bool,
}
