#if DIM == 2
    #import wgebra::sim2 as Pose
#else
    #import wgebra::sim3 as Pose
#endif
#import wgparry::ray as Ray
#import wgparry::projection as Proj

#define_import_path wgparry::cuboid

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


/// A box, defined by its half-extents (half-length alon geach dimension).
struct Cuboid {
    halfExtents: Vector
}

/// Projects a point on a box.
///
/// If the point is inside the box, the point itself is returned.
fn projectLocalPoint(box: Cuboid, pt: Vector) -> Vector {
    let mins = -box.halfExtents;
    let maxs = box.halfExtents;

    let mins_pt = mins - pt; // -hext - pt
    let pt_maxs = pt - maxs; // pt - hext
    let shift = max(mins_pt, Vector(0.0)) - max(pt_maxs, Vector(0.0));

    return pt + shift;
}

/// Projects a point on a transformed box.
///
/// If the point is inside the box, the point itself is returned.
fn projectPoint(box: Cuboid, pose: Transform, pt: Vector) -> Vector {
    let local_pt = Pose::invMulPt(pose, pt);
    return Pose::mulPt(pose, projectLocalPoint(box, local_pt));
}

/// Projects a point on the boundary of a box.
fn projectLocalPointOnBoundary(box: Cuboid, pt: Vector) -> Proj::ProjectionResult {
    let out_proj = projectLocalPoint(box, pt);

    // Projection if the point is inside the box.
    let pt_sgn_with_zero = sign(pt);
    // This the sign of pt, or -1 for components that were zero.
    // This bias is arbitrary (we could have picked +1), but we picked it so
    // it matches the bias that’s in parry.
    let pt_sgn = pt_sgn_with_zero + (abs(pt_sgn_with_zero) - Vector(1.0));
    let diff = box.halfExtents - pt_sgn * pt;

#if DIM == 2
    let pick_x = diff.x <= diff.y;
    let shift_x = Vector(diff.x * pt_sgn.x, 0.0);
    let shift_y = Vector(0.0, diff.y * pt_sgn.y);
    let pen_shift = select(shift_y, shift_x, pick_x);
#else
    let pick_x = diff.x <= diff.y && diff.x <= diff.z;
    let pick_y = diff.y <= diff.x && diff.y <= diff.z;
    let shift_x = Vector(diff.x * pt_sgn.x, 0.0, 0.0);
    let shift_y = Vector(0.0, diff.y * pt_sgn.y, 0.0);
    let shift_z = Vector(0.0, 0.0, diff.z * pt_sgn.z);
    let pen_shift = select(select(shift_z, shift_y, pick_y), shift_x, pick_x);
#endif
    let in_proj = pt + pen_shift;

    // Select between in and out proj.
    let is_inside = all(pt == out_proj);
    return Proj::ProjectionResult(select(out_proj, in_proj, is_inside), is_inside);
}

/// Project a point of a transformed box’s boundary.
///
/// If the point is inside of the box, it will be projected on its boundary but
/// `ProjectionResult::is_inside` will be set to `true`.
fn projectPointOnBoundary(box: Cuboid, pose: Transform, pt: Vector) -> Proj::ProjectionResult {
    let local_pt = Pose::invMulPt(pose, pt);
    var result = projectLocalPointOnBoundary(box, local_pt);
    result.point = Pose::mulPt(pose, result.point);
    return result;
}


// FIXME: ray.wgsl needs to support 2d/3d for these implementations to be commented-out.
///*
// * Ray casting.
// */
///// Casts a ray on a box.
/////
///// Returns a negative value if there is no hit.
///// If there is a hit, the result is a scalar `t >= 0` such that the hit point is equal to `ray.origin + t * ray.dir`.
//fn castLocalRay(box: Cuboid, ray: Ray::Ray, maxTimeOfImpact: f32) -> f32 {
//    let mins = -box.halfExtents;
//    let maxs = box.halfExtents;
//    let inter1 = (mins - ray.origin) / ray.dir;
//    let inter2 = (maxs - ray.origin) / ray.dir;
//
//    let vtmin = min(inter1, inter2);
//    let vtmax = max(inter1, inter2);
//
//#if DIM == 2
//    let tmin = max(max(vtmin.x, vtmin.y), 0.0);
//    let tmax = min(min(vtmax.x, vtmax.y), maxTimeOfImpact);
//#else
//    let tmin = max(max(max(vtmin.x, vtmin.y), vtmin.z), 0.0);
//    let tmax = min(min(min(vtmax.x, vtmax.y), vtmax.z), maxTimeOfImpact);
//#endif
//
//    if tmin > tmax || tmax < 0.0 {
//        return -1.0;
//    } else {
//        return tmin;
//    }
//}
//
///// Casts a ray on a transformed box.
/////
///// Returns a negative value if there is no hit.
///// If there is a hit, the result is a scalar `t >= 0` such that the hit point is equal to `ray.origin + t * ray.dir`.
//fn castRay(box: Cuboid, pose: Transform, ray: Ray::Ray, maxTimeOfImpact: f32) -> f32 {
//    let localRay = Ray::Ray(Pose::invMulPt(pose, ray.origin), Pose::invMulVec(pose, ray.dir));
//    return castLocalRay(box, localRay, maxTimeOfImpact);
//}
