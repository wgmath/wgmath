#if DIM == 2
    #import wgebra::sim2 as Pose
#else
    #import wgebra::sim3 as Pose
#endif
#import wgparry::ray as Ray
#import wgparry::projection as Proj
#import wgparry::segment as Seg

#define_import_path wgparry::capsule

/// A capsule, defined by its radius.
struct Capsule {
    /// The capsule’s principal axis.
    segment: Seg::Segment,
    /// The capsule’s radius.
    radius: f32,
}

fn orthonormal_basis3(v: vec3<f32>) -> array<vec3<f32>, 2> {
    // NOTE: not using `sign` because we don’t want the 0.0 case to return 0.0.
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

fn any_orthogonal_vector(v: Vector) -> Vector {
#if DIM == 2
    return vec2(v.y, -v.x);
#else
    return orthonormal_basis3(v)[0];
#endif
}

/// Projects a point on a capsule.
///
/// If the point is inside the capsule, the point itself is returned.
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

/// Projects a point on a transformed capsule.
///
/// If the point is inside the capsule, the point itself is returned.
fn projectPoint(capsule: Capsule, pose: Transform, pt: Vector) -> Vector {
    let localPt = Pose::invMulPt(pose, pt);
    return Pose::mulPt(pose, projectLocalPoint(capsule, localPt));
}


/// Projects a point on the boundary of a capsule.
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

/// Project a point of a transformed capsule’s boundary.
///
/// If the point is inside of the box, it will be projected on its boundary but
/// `ProjectionResult::is_inside` will be set to `true`.
fn projectPointOnBoundary(capsule: Capsule, pose: Transform, pt: Vector) -> Proj::ProjectionResult {
    let local_pt = Pose::invMulPt(pose, pt);
    var result = projectLocalPointOnBoundary(capsule, local_pt);
    result.point = Pose::mulPt(pose, result.point);
    return result;
}
