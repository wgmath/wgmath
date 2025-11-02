//! Separating Axis Theorem (SAT) Module
//!
//! This module implements the Separating Axis Theorem for cuboid-cuboid collision detection.
//! SAT tests potential separating axes to determine if two convex objects overlap.
//!
//! For cuboids, the potential separating axes are:
//! - Face normals of cuboid 1 (DIM axes)
//! - Face normals of cuboid 2 (DIM axes)
//! - Cross products of edge pairs (3x3=9 axes in 3D, not applicable in 2D)
//!
//! The algorithm finds the axis with maximum separation. If all separations are negative,
//! the cuboids are intersecting, and the axis with maximum penetration becomes the
//! contact normal.

#if DIM == 2
#import wgebra::sim2 as Pose
#else
#import wgebra::sim3 as Pose
#endif


#import wgparry::cuboid as Cuboid

#define_import_path wgparry::sat

/// Result of a separating axis test.
struct SeparatingAxis {
    /// Separation distance along the axis (negative = penetrating).
    separation: f32,
    /// The separating axis direction (normalized).
    axis: Vector,
}

/// Machine epsilon for floating-point comparisons.
const EPSILON: f32 = 1.1920929E-7;

#if DIM == 3
const DIM: u32 = 3;
#else
const DIM: u32 = 2;
#endif

#if DIM == 3
/// Computes the separation of two cuboids along `axis1`.
fn cuboid_cuboid_compute_separation_wrt_local_line(
    cuboid1: Cuboid::Cuboid,
    cuboid2: Cuboid::Cuboid,
    pos12: Transform,
    line_axis1: Vector,
) -> SeparatingAxis {
    let signum = select(-1.0, 1.0, dot(pos12.translation_scale.xyz, line_axis1) >= 0.0);
    let axis1 = line_axis1 * signum;
    let axis2 = Pose::invMulUnitVec(pos12, -axis1);
    let local_pt1 = Cuboid::local_support_point(cuboid1, axis1);
    let local_pt2 = Cuboid::local_support_point(cuboid2, axis2);
    let pt2 = Pose::mulPt(pos12, local_pt2);
    let separation = dot(pt2 - local_pt1, axis1);
    return SeparatingAxis(separation, axis1);
}

/// Finds the best separating edge between two cuboids.
///
/// All combinations of edges from both cuboids are taken into
/// account.
fn cuboid_cuboid_find_local_separating_edge_twoway(
    cuboid1: Cuboid::Cuboid,
    cuboid2: Cuboid::Cuboid,
    pos12: Transform,
) -> SeparatingAxis {
    var best_sep = SeparatingAxis(-1.0e10, vec3(0.0));

    let x2 = Pose::mulVec(pos12, vec3(1.0, 0.0, 0.0));
    let y2 = Pose::mulVec(pos12, vec3(0.0, 1.0, 0.0));
    let z2 = Pose::mulVec(pos12, vec3(0.0, 0.0, 1.0));

    // We have 3 * 3 = 9 axes to test.
    var axes = array(
        // Vector::{x, y ,z}().cross(y2)
        vec3(0.0, -x2.z, x2.y),
        vec3(x2.z, 0.0, -x2.x),
        vec3(-x2.y, x2.x, 0.0),
        // Vector::{x, y ,z}().cross(y2)
        vec3(0.0, -y2.z, y2.y),
        vec3(y2.z, 0.0, -y2.x),
        vec3(-y2.y, y2.x, 0.0),
        // Vector::{x, y ,z}().cross(y2)
        vec3(0.0, -z2.z, z2.y),
        vec3(z2.z, 0.0, -z2.x),
        vec3(-z2.y, z2.x, 0.0),
    );

    // TODO: unroll loop
    for (var i = 0u; i < 9u; i++) {
        let axis1 = axes[i];
        let norm1 = length(axis1);
        if norm1 > EPSILON {
            let sep = cuboid_cuboid_compute_separation_wrt_local_line(
                cuboid1,
                cuboid2,
                pos12,
                axis1 / norm1,
            );

            if sep.separation > best_sep.separation {
                best_sep = sep;
            }
        }
    }

    return best_sep;
}
#endif

/// Finds the best separating normal between two cuboids.
///
/// Only the normals from `cuboid1` are tested.
fn cuboid_cuboid_find_local_separating_normal_oneway(
    cuboid1: Cuboid::Cuboid,
    cuboid2: Cuboid::Cuboid,
    pos12: Transform,
) -> SeparatingAxis {
    var best_separation = -1.0e10;
    var best_dir = Vector(0.0);

    // TODO: unroll loop
    for (var i = 0u; i < DIM; i++) {
#if DIM == 2
        let sign = select(-1.0, 1.0, pos12.translation[i] >= 0.0);
#else
        let sign = select(-1.0, 1.0, pos12.translation_scale.xyz[i] >= 0.0);
#endif
        var axis1 = Vector(0.0);
        axis1[i] = sign;
        let axis2 = Pose::invMulUnitVec(pos12, -axis1);
        let local_pt2 = Cuboid::local_support_point(cuboid2, axis2);
        let pt2 = Pose::mulPt(pos12, local_pt2);
        let separation = pt2[i] * sign - cuboid1.halfExtents[i];

        if separation > best_separation {
            best_separation = separation;
            best_dir = axis1;
        }
    }

    return SeparatingAxis(best_separation, best_dir);
}
