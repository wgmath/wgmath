//! Capsule Shape Module
//!
//! This module provides geometric operations for capsules (swept spheres).
//! A capsule is defined by a line segment and a radius.

#if DIM == 2
    #import wgebra::sim2 as Pose
#else
    #import wgebra::sim3 as Pose
#endif
#import wgparry::ray as Ray
#import wgparry::projection as Proj
#import wgparry::segment as Seg
#import wgparry::polygonal_feature as Feat

#define_import_path wgparry::capsule

/// A capsule shape, defined by a central segment and a radius.
///
/// The capsule represents all points within `radius` distance from the segment.
/// This is equivalent to sweeping a ball along the segment's path.
struct Capsule {
    /// The capsule's principal axis (central line segment).
    /// The segment endpoints define the capsule's orientation and length.
    segment: Seg::Segment,
    /// The capsule's radius (distance from the central axis to the surface).
    /// Must be positive.
    radius: f32,
}

/// Computes an orthonormal basis from a single 3D vector.
///
/// Given a normalized vector v, this function computes two orthogonal vectors
/// that together with v form an orthonormal basis. This is useful for
/// constructing local coordinate frames.
///
/// Parameters:
/// - v: A normalized 3D vector
///
/// Returns: An array of two orthogonal unit vectors perpendicular to v.
fn orthonormal_basis3(v: vec3<f32>) -> array<vec3<f32>, 2> {
    // NOTE: not using `sign` because we don't want the 0.0 case to return 0.0.
    let sign = select(-1.0, 1.0, v.z >= 0.0);
    let a = -1.0 / (sign + v.z);
    let b = v.x * v.y * a;

    return array(
        vec3(
            1.0 + sign * v.x * v.x * a,
            sign * b,
            -sign * v.x,
        ),
        vec3(b, sign + v.y * v.y * a, -v.y),
    );
}

/// Finds an arbitrary vector orthogonal to the given vector.
///
/// This is a helper function used when a point lies exactly on the capsule's
/// axis and we need to pick an arbitrary direction for projection.
fn any_orthogonal_vector(v: Vector) -> Vector {
#if DIM == 2
    return vec2(v.y, -v.x);
#else
    return orthonormal_basis3(v)[0];
#endif
}

/// Projects a point onto a capsule in its local coordinate frame.
///
/// The algorithm first projects the point onto the central segment, then
/// projects radially onto the capsule surface if the point is outside.
///
/// Parameters:
/// - capsule: The capsule shape
/// - pt: The point to project (in the capsule's local frame)
///
/// Returns: The projected point (in the capsule's local frame)
fn projectLocalPoint(capsule: Capsule, pt: Vector) -> Vector {
    let proj_on_axis = Seg::projectLocalPoint(capsule.segment, pt);
    let dproj = pt - proj_on_axis;
    let dist_to_axis = length(dproj);

    // PERF: call `select` instead?
    if dist_to_axis > capsule.radius {
        return proj_on_axis + dproj * (capsule.radius / dist_to_axis);
    } else {
        return pt;
    }
}

/// Projects a point onto a transformed capsule in world space.
///
/// Parameters:
/// - capsule: The capsule shape
/// - pose: The capsule's world-space pose
/// - pt: The point to project (in world space)
///
/// Returns: The projected point (in world space)
fn projectPoint(capsule: Capsule, pose: Transform, pt: Vector) -> Vector {
    let localPt = Pose::invMulPt(pose, pt);
    return Pose::mulPt(pose, projectLocalPoint(capsule, localPt));
}


/// Projects a point onto the boundary (surface) of a capsule in its local frame.
///
/// This always projects onto the capsule's surface, even if the point is inside.
///
/// Parameters:
/// - capsule: The capsule shape
/// - pt: The point to project (in the capsule's local frame)
///
/// Returns: ProjectionResult containing the surface point and is_inside flag
///
/// Special case: If the point lies exactly on the capsule's central axis,
/// an arbitrary perpendicular direction is chosen for projection.
fn projectLocalPointOnBoundary(capsule: Capsule, pt: Vector) -> Proj::ProjectionResult {
    let proj_on_axis = Seg::projectLocalPoint(capsule.segment, pt);
    let dproj = pt - proj_on_axis;
    let dist_to_axis = length(dproj);

    if dist_to_axis > 0.0 {
        let is_inside = dist_to_axis <= capsule.radius;
        return Proj::ProjectionResult(proj_on_axis + dproj * (capsule.radius / dist_to_axis), is_inside);
    } else {
        // Very rare occurence: the point lies on the capsule’s axis exactly.
        // Pick an arbitrary projection direction along an axis orthogonal to the principal axis.
        let axis_seg = capsule.segment.b - capsule.segment.a;
        let axis_len = length(axis_seg);
        let proj_dir = any_orthogonal_vector(axis_seg / axis_len);
        return Proj::ProjectionResult(proj_on_axis + proj_dir * capsule.radius, true);
    }
}

/// Projects a point onto the boundary of a transformed capsule in world space.
///
/// Parameters:
/// - capsule: The capsule shape
/// - pose: The capsule's world-space pose
/// - pt: The point to project (in world space)
///
/// Returns: ProjectionResult with point in world space and is_inside flag
fn projectPointOnBoundary(capsule: Capsule, pose: Transform, pt: Vector) -> Proj::ProjectionResult {
    let local_pt = Pose::invMulPt(pose, pt);
    var result = projectLocalPointOnBoundary(capsule, local_pt);
    result.point = Pose::mulPt(pose, result.point);
    return result;
}


fn local_support_point(shape: Capsule, dir: Vector) -> Vector {
    let seg_dir = shape.segment.b - shape.segment.a;
    let endpoint = select(shape.segment.a, shape.segment.b, dot(seg_dir, dir) >= 0.0);

    if shape.radius == 0.0 {
        return endpoint;
    }

    let dir_len = length(dir);
    let normal = select(Vector(0.0, 1.0, 0.0), dir / dir_len, dir_len != 0.0);
    return endpoint + normal * shape.radius;
}

fn support_face(shape: Capsule, dir: Vector) -> Feat::PolygonalFeature {
    var result = Feat::PolygonalFeature();
    if shape.radius == 0.0 {
        result.vertices[0] = shape.segment.a;
        result.vertices[1] = shape.segment.b;
        result.num_vertices = 2;
    } else {
        let seg_dir = shape.segment.b - shape.segment.a;
        if abs(dot(seg_dir, dir)) <= 1.0e-6 {
            result.vertices[0] = shape.segment.a;
            result.vertices[1] = shape.segment.b;
            result.num_vertices = 2;
        } else {
            let endpoint = select(shape.segment.a, shape.segment.b, dot(seg_dir, dir) >= 0.0);
            result.vertices[0] = endpoint + dir * (shape.radius / length(dir));
            result.num_vertices = 1;
        }
    }

    return result;
}