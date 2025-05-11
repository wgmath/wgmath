#if DIM == 2
    #import wgebra::sim2 as Pose
#else
    #import wgebra::sim3 as Pose
#endif
#import wgparry::ray as Ray
#import wgparry::projection as Proj
#import wgparry::segment as Seg

#define_import_path wgparry::cone

/// A cone, defined by its radius.
struct Cone {
    /// The cone’s principal axis.
    half_height: f32,
    /// The cone’s radius.
    radius: f32,
}

/// Projects a point on a cone.
///
/// If the point is inside the cone, the point itself is returned.
fn projectLocalPoint(cone: Cone, pt: Vector) -> Vector {
    // Project on the basis.
    let planar_dist_from_basis_center = length(pt.xz);
    let dir_from_basis_center = select(
        vec2(1.0, 0.0),
        pt.xz / planar_dist_from_basis_center,
        planar_dist_from_basis_center > 0.0
    );

    let projection_on_basis = vec3(pt.x, -cone.half_height, pt.z);

    if pt.y < -cone.half_height && planar_dist_from_basis_center <= cone.radius {
        // The projection is on the basis.
        return projection_on_basis;
    }

    // Project on the basis circle.
    let proj2d = dir_from_basis_center * cone.radius;
    let projection_on_basis_circle = vec3(proj2d[0], -cone.half_height, proj2d[1]);

    // Project on the conic side.
    // TODO: we could solve this in 2D using the plane passing through the cone axis and the conic_side_segment to save some computation.
    let apex_point = vec3(0.0, cone.half_height, 0.0);
    let conic_side_segment = Seg::Segment(apex_point, projection_on_basis_circle);
    let conic_side_segment_dir = conic_side_segment.b - conic_side_segment.a;
    let proj = Seg::projectLocalPoint(conic_side_segment, pt);

    let apex_to_basis_center = vec3(0.0, -2.0 * cone.half_height, 0.0);

    // Now determine if the point is inside of the cone.
    if pt.y >= -cone.half_height
        && pt.y <= cone.half_height
        && dot(
               cross(conic_side_segment_dir, pt - apex_point),
               cross(conic_side_segment_dir, apex_to_basis_center)
           ) >= 0.0
    {
        // We are inside of the cone.
        return pt;
    } else {
        // We are outside of the cone, return the computed segment projection.
        return proj;
    }
}

/// Projects a point on a transformed cone.
///
/// If the point is inside the cone, the point itself is returned.
fn projectPoint(cone: Cone, pose: Transform, pt: Vector) -> Vector {
    let localPt = Pose::invMulPt(pose, pt);
    return Pose::mulPt(pose, projectLocalPoint(cone, localPt));
}


/// Projects a point on the boundary of a cone.
fn projectLocalPointOnBoundary(cone: Cone, pt: Vector) -> Proj::ProjectionResult {
    // Project on the basis.
    let planar_dist_from_basis_center = length(pt.xz);
    let dir_from_basis_center = select(
        vec2(1.0, 0.0),
        pt.xz / planar_dist_from_basis_center,
        planar_dist_from_basis_center > 0.0
    );

    let projection_on_basis = vec3(pt.x, -cone.half_height, pt.z);

    if pt.y < -cone.half_height && planar_dist_from_basis_center <= cone.radius {
        // The projection is on the basis.
        return Proj::ProjectionResult(projection_on_basis, false);
    }

    // Project on the basis circle.
    let proj2d = dir_from_basis_center * cone.radius;
    let projection_on_basis_circle = vec3(proj2d[0], -cone.half_height, proj2d[1]);

    // Project on the conic side.
    // TODO: we could solve this in 2D using the plane passing through the cone axis and the conic_side_segment to save some computation.
    let apex_point = vec3(0.0, cone.half_height, 0.0);
    let conic_side_segment = Seg::Segment(apex_point, projection_on_basis_circle);
    let conic_side_segment_dir = conic_side_segment.b - conic_side_segment.a;
    let proj = Seg::projectLocalPoint(conic_side_segment, pt);

    let apex_to_basis_center = vec3(0.0, -2.0 * cone.half_height, 0.0);

    // Now determine if the point is inside of the cone.
    if pt.y >= -cone.half_height
        && pt.y <= cone.half_height
        && dot(
               cross(conic_side_segment_dir, pt - apex_point),
               cross(conic_side_segment_dir, apex_to_basis_center)
           ) >= 0.0
    {
        // We are inside of the cone, so the correct projection is
        // either on the basis of the cone, or on the conic side.
        let pt_to_proj = proj - pt;
        let pt_to_basis_proj = projection_on_basis - pt;
        if dot(pt_to_proj, pt_to_proj) > dot(pt_to_basis_proj, pt_to_basis_proj) {
            return Proj::ProjectionResult(projection_on_basis, true);
        } else {
            return Proj::ProjectionResult(proj, true);
        }
    } else {
        // We are outside of the cone, return the computed segment projection as-is.
        return Proj::ProjectionResult(proj, false);
    }
}

/// Project a point of a transformed cone’s boundary.
///
/// If the point is inside of the box, it will be projected on its boundary but
/// `ProjectionResult::is_inside` will be set to `true`.
fn projectPointOnBoundary(cone: Cone, pose: Transform, pt: Vector) -> Proj::ProjectionResult {
    let local_pt = Pose::invMulPt(pose, pt);
    var result = projectLocalPointOnBoundary(cone, local_pt);
    result.point = Pose::mulPt(pose, result.point);
    return result;
}
