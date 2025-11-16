//! Contact Computation Module
//!
//! This module provides high-level contact computation functions for various shape pairs.
//! Each function implements the narrow-phase collision detection algorithm specific to
//! that pair of shape types.
//!
//! Implemented shape pairs:
//! - Ball-Ball: Analytical distance-based computation
//! - Cuboid-Ball: Point projection onto cuboid boundary
//! - Ball-Cuboid: Inverse of cuboid-ball with coordinate transform
//! - Cuboid-Cuboid: SAT + polygonal feature clipping
//!
//! Algorithm patterns:
//! - Simple shapes (ball-ball): Direct analytical formulas
//! - Mixed shapes (cuboid-ball): Point projection + distance test
//! - Complex shapes (cuboid-cuboid): Multi-stage SAT + clipping + reduction
//!
//! All functions operate in the local frame of the first shape, with the second
//! shape's pose provided as a relative transform.

#if DIM == 2
    #import wgebra::sim2 as Pose
#else
    #import wgebra::sim3 as Pose
#endif

#import wgparry::ball as Ball
#import wgparry::cuboid as Cuboid
#import wgparry::sat as Sat
#import wgparry::polygonal_feature as PolygonalFeature
#import wgparry::contact_manifold as Manifold
#import wgparry::queries::contact_pfm_pfm_generic as PfmPfm

#define_import_path wgparry::contact


/// Contact manifold with collider pair indices for solver integration.
///
/// This structure extends ContactManifold with the collider indices,
/// allowing the physics solver to identify which bodies are in contact.
struct IndexedManifold {
    /// The contact information.
    contact: Manifold::ContactManifold,
    /// Collider pair that resulted in this contact.
    colliders: vec2<u32>,
}

/// Computes the contact between two balls.
fn ball_ball(pose12: Transform, ball1: Ball::Ball, ball2: Ball::Ball) -> Manifold::ContactManifold {
    let r1 = ball1.radius;
    let r2 = ball2.radius;

#if DIM == 2
    let center2_1 = pose12.translation;
    var normal1 = vec2(0.0, 1.0);
#else
    let center2_1 = pose12.translation_scale.xyz;
    var normal1 = vec3(0.0, 1.0, 0.0);
#endif

    let distance = length(center2_1);
    let sum_radius = r1 + r2;

    if distance != 0.0 {
        normal1 = center2_1 / distance;
    }

    let point1 = normal1 * r1;

    return Manifold::single_point(point1, distance - sum_radius, normal1);
}

// TODO: generalize to all convex shapes implementing point projection.
fn cuboid_ball(pose12: Transform, cuboid1: Cuboid::Cuboid, ball2: Ball::Ball) -> Manifold::ContactManifold {
#if DIM == 2
    let center2_1 = pose12.translation;
    var y = vec2(0.0, 1.0);
#else
    let center2_1 = pose12.translation_scale.xyz;
    var y = vec3(0.0, 1.0, 0.0);
#endif

    let proj = Cuboid::projectLocalPointOnBoundary(cuboid1, center2_1);
    let proj_vec = center2_1 - proj.point;
    var dist = length(proj_vec);
    var normal1 = select(y, proj_vec / dist, dist != 0.0);

    if proj.is_inside {
        normal1 = -normal1;
        dist = -dist;
    }

    return Manifold::single_point(proj.point, dist - ball2.radius, normal1);
}

fn ball_cuboid(pose12: Transform, ball1: Ball::Ball, cuboid2: Cuboid::Cuboid) -> Manifold::ContactManifold {
    let pose21 = Pose::inv(pose12);
    var result = cuboid_ball(pose21, cuboid2, ball1);
    let normal1 = -Pose::mulUnitVec(pose12, result.normal_a);
    result.points_a[0].pt = normal1 * ball1.radius;
    result.normal_a = normal1;
    return result;
}

fn cuboid_cuboid(pose12: Transform, cuboid1: Cuboid::Cuboid, cuboid2: Cuboid::Cuboid, prediction: f32) -> Manifold::ContactManifold {
    return PfmPfm::contact_manifold_pfm_pfm(pose12, cuboid1, 0.1, cuboid2, 0.1, prediction);
}

fn cuboid_cuboid_sat(pose12: Transform, cuboid1: Cuboid::Cuboid, cuboid2: Cuboid::Cuboid, prediction: f32) -> Manifold::ContactManifold {
    let pose21 = Pose::inv(pose12);

    /*
     *
     * Point-Face
     *
     */
    let sep1 = Sat::cuboid_cuboid_find_local_separating_normal_oneway(cuboid1, cuboid2, pose12);

      // TODO PERF: support the prediction early-exit.
//    if sep1.0 > prediction {
//        manifold.clear();
//        return;
//    }

    let sep2 = Sat::cuboid_cuboid_find_local_separating_normal_oneway(cuboid2, cuboid1, pose21);

      // TODO PERF: support the prediction early-exit.
//    if sep2.0 > prediction {
//        manifold.clear();
//        return;
//    }

    /*
     *
     * Edge-Edge cases
     *
     */
#if DIM == 2
    let sep3 = Sat::SeparatingAxis(-1.0e10, vec2(1.0, 0.0)); // This case does not exist in 2D.
#else
    let sep3 = Sat::cuboid_cuboid_find_local_separating_edge_twoway(cuboid1, cuboid2, pose12);

      // TODO PERF: support the prediction early-exit.
//    if sep3.0 > prediction {
//        manifold.clear();
//        return;
//    }
#endif

    /*
     *
     * Select the best combination of features
     * and get the polygons to clip.
     *
     */
    var best_sep = sep1;

    if sep2.separation > sep1.separation && sep2.separation > sep3.separation {
        best_sep = Sat::SeparatingAxis(sep2.separation, Pose::mulUnitVec(pose12, -sep2.axis));
    } else if sep3.separation > sep1.separation {
        best_sep = sep3;
    }

    let local_n2 = Pose::mulUnitVec(pose21, -best_sep.axis);

    // Now the reference feature is from `cuboid1` and the best separation is `best_sep`.
    // Everything must be expressed in the local-space of `cuboid1` for contact clipping.
    let face1 = Cuboid::support_face(cuboid1, best_sep.axis);
    let face2 = Cuboid::support_face(cuboid2, local_n2);
    var manifold = PolygonalFeature::contacts(
        pose12,
        pose21,
        best_sep.axis,
        local_n2,
        face1,
        face2,
        prediction,
        false
    );
    manifold.normal_a = best_sep.axis;
    return manifold;
}