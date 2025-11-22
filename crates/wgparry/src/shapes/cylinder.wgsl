//! Cylinder Shape Module (3D only)
//!
//! This module provides geometric operations for cylinders.
//! A cylinder is defined by its half-height and radius.
//!
//! The cylinder is oriented along the Y axis with:
//! - Top at y = +half_height.
//! - Bottom at y = -half_height.
//! - Circular cross-section in the XZ plane.

#if DIM == 2
    #import wgebra::sim2 as Pose
#else
    #import wgebra::sim3 as Pose
#endif
#import wgparry::ray as Ray
#import wgparry::projection as Proj
#import wgparry::polygonal_feature as Feat

#define_import_path wgparry::cylinder

/// A cylinder shape with circular cross-section (3D only).
///
/// The cylinder is aligned with the Y axis, extending from
/// y = -half_height to y = +half_height.
struct Cylinder {
    /// Half of the cylinder's height.
    half_height: f32,
    /// Radius of the circular cross-section.
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

fn support_point(cylinder: Cylinder, pose: Transform, axis: Vector) -> Vector {
    let local_axis = Pose::invMulVec(pose, axis);
    let local_pt = local_support_point(cylinder, local_axis);
    return Pose::mulPt(pose, local_pt);
}

fn local_support_point(cylinder: Cylinder, dir: Vector) -> Vector {
    var vres = dir;
    vres.y = 0.0;

    let planar_dir_len = length(vres);
    let factor = select(0.0, cylinder.radius / planar_dir_len, planar_dir_len != 0.0);
    vres *= factor;
    vres.y = select(-cylinder.half_height, cylinder.half_height, dir.y >= 0.0);
    return vres;
}

fn support_face(cylinder: Cylinder, dir: Vector) -> Feat::PolygonalFeature {
    var result = Feat::PolygonalFeature();

    var dir2 = vec2(dir.x, dir.z);
    let dir2_len = length(dir2);
    if dir2_len < Proj::EPSILON.x {
        dir2 = vec2(1.0, 0.0);
    } else {
        dir2 /= dir2_len;
    }

    if abs(dir.y) < 0.5 {
        // We return a segment lying on the cylinder's curved part.
        result.vertices[0] = vec3(
            dir2.x * cylinder.radius,
            -cylinder.half_height,
            dir2.y * cylinder.radius,
        );
        result.vertices[1] =
            vec3(dir2.x * cylinder.radius, cylinder.half_height, dir2.y * cylinder.radius);
        result.num_vertices = 2;
    } else {
        // We return a square approximation of the cylinder cap.
        let y = select(-cylinder.half_height, cylinder.half_height, dir.y >= 0.0);
        result.vertices[0] = vec3(dir2.x * cylinder.radius, y, dir2.y * cylinder.radius);
        result.vertices[1] = vec3(-dir2.y * cylinder.radius, y, dir2.x * cylinder.radius);
        result.vertices[2] = vec3(-dir2.x * cylinder.radius, y, -dir2.y * cylinder.radius);
        result.vertices[3] = vec3(dir2.y * cylinder.radius, y, -dir2.x * cylinder.radius);
        result.num_vertices = 4;
    }

    return result;
}