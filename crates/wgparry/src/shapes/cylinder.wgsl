#if DIM == 2
    #import wgebra::sim2 as Pose
#else
    #import wgebra::sim3 as Pose
#endif
#import wgparry::ray as Ray
#import wgparry::projection as Proj

#define_import_path wgparry::cylinder

/// A cylinder, defined by its radius.
struct Cylinder {
    /// The cylinder’s principal axis.
    half_height: f32,
    /// The cylinder’s radius.
    radius: f32,
}

/// Projects a point on a cylinder.
///
/// If the point is inside the cylinder, the point itself is returned.
fn projectLocalPoint(cylinder: Cylinder, pt: Vector) -> Vector {

    // Project on the basis.
    let planar_dist_from_basis_center = length(pt.xz);
    let dir_from_basis_center = select(
        vec2(1.0, 0.0),
        pt.xz / planar_dist_from_basis_center,
        planar_dist_from_basis_center > 0.0
    );

    let proj2d = dir_from_basis_center * cylinder.radius;

    // PERF: reduce branching
    if pt.y >= -cylinder.half_height
        && pt.y <= cylinder.half_height
        && planar_dist_from_basis_center <= cylinder.radius
    {
        return pt;
    } else {
        // The point is outside of the cylinder.
        if pt.y > cylinder.half_height {
            if planar_dist_from_basis_center <= cylinder.radius {
                return vec3(pt.x, cylinder.half_height, pt.z);
            } else {
                return vec3(proj2d[0], cylinder.half_height, proj2d[1]);
            }
        } else if pt.y < -cylinder.half_height {
            // Project on the bottom plane or the bottom circle.
            if planar_dist_from_basis_center <= cylinder.radius {
                return vec3(pt.x, -cylinder.half_height, pt.z);
            } else {
                return vec3(proj2d[0], -cylinder.half_height, proj2d[1]);
            }
        } else {
            // Project on the side.
            return vec3(proj2d[0], pt.y, proj2d[1]);
        }
    }
}

/// Projects a point on a transformed cylinder.
///
/// If the point is inside the cylinder, the point itself is returned.
fn projectPoint(cylinder: Cylinder, pose: Transform, pt: Vector) -> Vector {
    let localPt = Pose::invMulPt(pose, pt);
    return Pose::mulPt(pose, projectLocalPoint(cylinder, localPt));
}


/// Projects a point on the boundary of a cylinder.
fn projectLocalPointOnBoundary(cylinder: Cylinder, pt: Vector) -> Proj::ProjectionResult {
    // Project on the basis.
    let planar_dist_from_basis_center = length(pt.xz);
    let dir_from_basis_center = select(
        vec2(1.0, 0.0),
        pt.xz / planar_dist_from_basis_center,
        planar_dist_from_basis_center > 0.0
    );

    let proj2d = dir_from_basis_center * cylinder.radius;

    // PERF: reduce branching
    if pt.y >= -cylinder.half_height
        && pt.y <= cylinder.half_height
        && planar_dist_from_basis_center <= cylinder.radius
    {
        // The point is inside of the cylinder.
        let dist_to_top = cylinder.half_height - pt.y;
        let dist_to_bottom = pt.y - (-cylinder.half_height);
        let dist_to_side = cylinder.radius - planar_dist_from_basis_center;

        if dist_to_top < dist_to_bottom && dist_to_top < dist_to_side {
            let projection_on_top = vec3(pt.x, cylinder.half_height, pt.z);
            return Proj::ProjectionResult(projection_on_top, true);
        } else if dist_to_bottom < dist_to_top && dist_to_bottom < dist_to_side {
            let projection_on_bottom =
                vec3(pt.x, -cylinder.half_height, pt.z);
            return Proj::ProjectionResult(projection_on_bottom, true);
        } else {
            let projection_on_side = vec3(proj2d[0], pt.y, proj2d[1]);
            return Proj::ProjectionResult(projection_on_side, true);
        }
    } else {
        // The point is outside of the cylinder.
        if pt.y > cylinder.half_height {
            if planar_dist_from_basis_center <= cylinder.radius {
                let projection_on_top = vec3(pt.x, cylinder.half_height, pt.z);
                return Proj::ProjectionResult(projection_on_top, false);
            } else {
                let projection_on_top_circle =
                    vec3(proj2d[0], cylinder.half_height, proj2d[1]);
                return Proj::ProjectionResult(projection_on_top_circle, false);
            }
        } else if pt.y < -cylinder.half_height {
            // Project on the bottom plane or the bottom circle.
            if planar_dist_from_basis_center <= cylinder.radius {
                let projection_on_bottom =
                    vec3(pt.x, -cylinder.half_height, pt.z);
                return Proj::ProjectionResult(projection_on_bottom, false);
            } else {
                let projection_on_bottom_circle =
                    vec3(proj2d[0], -cylinder.half_height, proj2d[1]);
                return Proj::ProjectionResult(projection_on_bottom_circle, false);
            }
        } else {
            // Project on the side.
            let projection_on_side = vec3(proj2d[0], pt.y, proj2d[1]);
            return Proj::ProjectionResult(projection_on_side, false);
        }
    }
}

/// Project a point of a transformed cylinder’s boundary.
///
/// If the point is inside of the box, it will be projected on its boundary but
/// `ProjectionResult::is_inside` will be set to `true`.
fn projectPointOnBoundary(cylinder: Cylinder, pose: Transform, pt: Vector) -> Proj::ProjectionResult {
    let local_pt = Pose::invMulPt(pose, pt);
    var result = projectLocalPointOnBoundary(cylinder, local_pt);
    result.point = Pose::mulPt(pose, result.point);
    return result;
}
