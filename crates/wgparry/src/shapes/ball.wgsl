//! Ball (sphere/circle) Shape Module
//!
//! This module provides geometric operations for balls (spheres in 3D, circles in 2D).
//! A ball is defined by a single radius parameter and represents the simplest convex shape.
//!
//! Dimension-specific behavior:
//! - 2D: Ball represents a circle with uniform radius.
//! - 3D: Ball represents a sphere with uniform radius.

#if DIM == 2
    #import wgebra::sim2 as Pose
#else
    #import wgebra::sim3 as Pose
#endif
#import wgparry::ray as Ray
#import wgparry::projection as Proj

#define_import_path wgparry::ball


/// A ball shape, defined by its radius.
///
/// This represents a sphere in 3D or a circle in 2D, centered at the origin
/// in its local coordinate frame.
struct Ball {
    /// The ball's radius. Must be positive.
    radius: f32,
}

/*
/// Casts a ray on a ball.
///
/// Returns a negative value if there is no hit.
/// If there is a hit, the result is a scalar `t >= 0` such that the hit point is equal to `ray.origin + t * ray.dir`.
fn castLocalRay(ball: Ball, ray: Ray::Ray, maxTimeOfImpact: f32) -> f32 {
    // Ray origin relative to the ball’s center. It’s the origin itself since it’s in the ball’s local frame.
    let dcenter = ray.origin;
    let a = dot(ray.dir, ray.dir);
    let b = dot(dcenter, ray.dir);
    let c = dot(dcenter, dcenter) - ball.radius * ball.radius;
    let delta = b * b - a * c;
    let t = -b - sqrt(delta);

    if (c > 0.0 && (b > 0.0 || a == 0.0)) || delta < 0.0 || t > maxTimeOfImpact * a {
        // No hit.
        return -1.0;
    } else if a == 0.0 {
        // Dir is zero but the ray started inside the ball.
        return 0.0;
    } else {
        // Hit. If t <= 0, the origin is inside the ball.
        return max(t / a, 0.0);
    }
}

/// Casts a ray on a transformed ball.
///
/// Returns a negative value if there is no hit.
/// If there is a hit, the result is a scalar `t >= 0` such that the hit point is equal to `ray.origin + t * ray.dir`.
fn castRay(ball: Ball, pose: Transform, ray: Ray::Ray, maxTimeOfImpact: f32) -> f32 {
    let localRay = Ray::Ray(Pose::invMulPt(pose, ray.origin), Pose::invMulVec(pose, ray.dir));
    return castLocalRay(ball, localRay, maxTimeOfImpact);
}
*/

/// Projects a point onto a ball in its local coordinate frame.
///
/// This function finds the closest point on or inside the ball to the given point.
/// If the point is already inside the ball, it is returned unchanged.
/// If the point is outside, it is projected onto the ball's surface along the
/// direction from the center to the point.
///
/// Parameters:
/// - ball: The ball shape
/// - pt: The point to project (in the ball's local frame)
///
/// Returns: The projected point (in the ball's local frame)
fn projectLocalPoint(ball: Ball, pt: Vector) -> Vector {
    let dist = length(pt);

    if dist >= ball.radius {
        // The point is outside the ball.
        return pt * (ball.radius / dist);
    } else {
        // The point is inside the ball.
        return pt;
    }
}

/// Projects a point onto a transformed ball in world space.
///
/// This is a convenience wrapper that transforms the point to the ball's local
/// frame, performs the projection, then transforms the result back to world space.
///
/// Parameters:
/// - ball: The ball shape
/// - pose: The ball's world-space pose (position, rotation, scale)
/// - pt: The point to project (in world space)
///
/// Returns: The projected point (in world space)
fn projectPoint(ball: Ball, pose: Transform, pt: Vector) -> Vector {
    let localPt = Pose::invMulPt(pose, pt);
    return Pose::mulPt(pose, projectLocalPoint(ball, localPt));
}


/// Projects a point onto the boundary (surface) of a ball in its local frame.
///
/// Unlike projectLocalPoint, this always projects onto the ball's surface, even
/// if the point is inside. This is useful for finding the closest surface point.
///
/// Parameters:
/// - ball: The ball shape
/// - pt: The point to project (in the ball's local frame)
///
/// Returns: ProjectionResult containing:
/// - point: The projected point on the ball's surface
/// - is_inside: true if the original point was inside the ball
///
/// Special case: If the point is exactly at the center (distance == 0),
/// a fallback direction is used to avoid division by zero.
fn projectLocalPointOnBoundary(ball: Ball, pt: Vector) -> Proj::ProjectionResult {
    let dist = length(pt);
#if DIM == 2
    let fallback = vec2(0.0, ball.radius);
#else
    let fallback = vec3(0.0, ball.radius, 0.0);
#endif

    let projected_point =
        select(fallback, pt * (ball.radius / dist), dist != 0.0);
    let is_inside = dist <= ball.radius;

    return Proj::ProjectionResult(projected_point, is_inside);
}

/// Projects a point onto the boundary of a transformed ball in world space.
///
/// This is a convenience wrapper that transforms the point to the ball's local
/// frame, performs boundary projection, then transforms the result back to world space.
///
/// Parameters:
/// - ball: The ball shape
/// - pose: The ball's world-space pose
/// - pt: The point to project (in world space)
///
/// Returns: ProjectionResult with point in world space and is_inside flag
fn projectPointOnBoundary(ball: Ball, pose: Transform, pt: Vector) -> Proj::ProjectionResult {
    let local_pt = Pose::invMulPt(pose, pt);
    var result = projectLocalPointOnBoundary(ball, local_pt);
    result.point = Pose::mulPt(pose, result.point);
    return result;
}
