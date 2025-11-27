#define_import_path wgparry::gjk::cso_point


const FLT_EPS: f32 = 1.0e-7;
const EPS_TOL: f32 = 1.0e-6;

/// A point of a Configuration-Space Obstacle.
///
/// A Configuration-Space Obstacle (CSO) is the result of the
/// Minkowski Difference of two solids. In other words, each of its
/// points correspond to the difference of two point, each belonging
/// to a different solid.
struct CsoPoint {
    /// The point on the CSO. This is equal to `self.orig1 - self.orig2`, unless this CsoPoint
    /// has been translated with self.translate.
    point: Vector,
    /// The original point on the first shape used to compute `self.point`.
    orig_a: Vector,
    /// The original point on the second shape used to compute `self.point`.
    orig_b: Vector,
}

/// Initializes a CSO point with `orig1 - orig2`.
fn from_points(orig1: Vector, orig2: Vector) -> CsoPoint {
    return CsoPoint(orig1 - orig2, orig1, orig2);
}

/// Initializes a CSO point with all information provided.
///
/// It is assumed, but not checked, that `point == orig1 - orig2`.
fn from_parts(point: Vector, orig1: Vector, orig2: Vector) -> CsoPoint {
    return CsoPoint(point, orig1, orig2);
}

/// Initializes a CSO point where both original points are equal.
fn single_point(point: Vector) -> CsoPoint {
    return CsoPoint(point, point, Vector());
}

/// CSO point where all components are set to zero.
fn origin() -> CsoPoint {
    return CsoPoint(Vector(), Vector(), Vector());
}

