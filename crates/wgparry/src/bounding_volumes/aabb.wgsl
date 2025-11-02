//! Axis-Aligned Bounding Box (AABB) Module
//!
//! This module provides AABB computation and intersection testing for all shape types.
//! AABBs are used extensively in broad-phase collision detection to quickly prune
//! non-intersecting pairs.
//!
//! Key operations:
//! - from_shape: Computes tight AABB for any shape type
//! - check_intersection: Fast AABB-AABB overlap test
//! - merge: Computes bounding AABB of two AABBs

#define_import_path wgparry::bounding_volumes::aabb

#if DIM == 2
    #import wgebra::sim2 as Pose;
    #import wgebra::rot2 as Rot;
#else
    #import wgebra::sim3 as Pose;
    #import wgebra::quat as Rot;
#endif

#import wgparry::shape as Shape;

/// An axis-aligned bounding box defined by minimum and maximum corners.
struct Aabb {
    /// Minimum corner (smallest coordinates along each axis).
    mins: Vector,
    /// Maximum corner (largest coordinates along each axis).
    maxs: Vector
}

/// Creates an AABB from a transformed shape.
fn from_shape(pose: Transform, shape: Shape::Shape) -> Aabb {
    let ty = Shape::shape_type(shape);
    if ty == Shape::SHAPE_TYPE_BALL {
        let ball = Shape::to_ball(shape);
#if DIM == 2
        let tra = pose.translation;
        let rad = ball.radius * pose.scale;
#else
        let tra = pose.translation_scale.xyz;
        let rad = ball.radius * pose.translation_scale.w;
#endif

        return Aabb(
            tra - Vector(rad),
            tra + Vector(rad)
        );
    }

    if ty == Shape::SHAPE_TYPE_CUBOID {
        let cuboid = Shape::to_cuboid(shape);
        let rotmat = Rot::toMatrix(pose.rotation);

#if DIM == 2
        // TODO: write a utility for abs(mat2x2).
        let hext = mat2x2(abs(rotmat[0]), abs(rotmat[1])) * (cuboid.halfExtents * pose.scale);
        let center = pose.translation;
#else
        // TODO: write a utility for abs(mat3x3).
        let hext = mat3x3(abs(rotmat[0]), abs(rotmat[1]), abs(rotmat[2])) * (cuboid.halfExtents * pose.translation_scale.w);
        let center = pose.translation_scale.xyz;
#endif

        return Aabb(
            center - hext,
            center + hext
        );
    }

    if ty == Shape::SHAPE_TYPE_CAPSULE {
        let capsule = Shape::to_capsule(shape);
        // FIXME
        return Aabb();
    }

#if DIM == 3
    if ty == Shape::SHAPE_TYPE_CONE {
        let cone = Shape::to_cone(shape);
        // FIXME
        return Aabb();
    }

    if ty == Shape::SHAPE_TYPE_CYLINDER {
        let cylinder = Shape::to_cylinder(shape);
        // FIXME
        return Aabb();
    }
#endif

    // TODO: not implemented.
    return Aabb();
}

/// Are the two AABBs intersecting?
fn check_intersection(aabb1: Aabb, aabb2: Aabb) -> bool {
    return !(any(aabb2.maxs < aabb1.mins) || any(aabb1.maxs < aabb2.mins));
}

/// Merge two AABBs into a single one that tightly encloses both inputs.
fn merge(aabb1: Aabb, aabb2: Aabb) -> Aabb {
    return Aabb(min(aabb1.mins, aabb2.mins), max(aabb1.maxs, aabb2.maxs));
}