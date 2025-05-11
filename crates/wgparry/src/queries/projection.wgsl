#define_import_path wgparry::projection

/// The result of a point projection.
struct ProjectionResult {
    /// The point’s projection on the shape.
    /// This can be equal to the original point if the point was inside
    /// of the shape and the projection function doesn’t always project
    /// on the boundary.
    point: Vector,
    /// Is the point inside of the shape?
    is_inside: bool,
}
