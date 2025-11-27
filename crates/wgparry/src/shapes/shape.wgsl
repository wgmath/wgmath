//! Generic Shape Enumeration Module
//!
//! This module provides a unified interface for working with multiple shape types
//! through runtime polymorphism. Shapes are stored in a compact format using
//! two vec4 fields, with the shape type encoded in the last component.
//!
//! Shape Type Encoding:
//! - BALL (0): vec4(radius, _, _, type), vec4(_, _, _, _)
//! - CUBOID (1): vec4(hx, hy, hz, type), vec4(_, _, _, _)
//! - CAPSULE (2): vec4(ax, ay, az, type), vec4(bx, by, bz, radius)
//! - CONE (3): vec4(half_height, radius, _, type), vec4(_, _, _, _) [3D only]
//! - CYLINDER (4): vec4(half_height, radius, _, type), vec4(_, _, _, _) [3D only]
//! - POLYLINE (5): Not yet implemented
//! - TRIMESH (6): Not yet implemented

#if DIM == 2
    #import wgebra::sim2 as Pose
#else
    #import wgebra::sim3 as Pose
#endif
#import wgparry::projection as Proj
#import wgparry::capsule as Cap;
#import wgparry::ball as Bal;
#import wgparry::cuboid as Cub;
#import wgparry::segment as Seg;
#import wgparry::cylinder as Cyl;
#import wgparry::cone as Con;
#import wgparry::polygonal_feature as Feat
#import wgparry::convex as Convex;
#import wgparry::trimesh as TriMesh;
#import wgparry::polyline as Polyline;
#import wgparry::triangle as Tri;
#import wgparry::bounding_volumes::aabb as Aabb;

#define_import_path wgparry::shape

/// Shape type constants for runtime type identification
const SHAPE_TYPE_BALL: u32 = 0;
const SHAPE_TYPE_CUBOID: u32 = 1;
const SHAPE_TYPE_CAPSULE: u32 = 2;
const SHAPE_TYPE_CONE: u32 = 3;
const SHAPE_TYPE_CYLINDER: u32 = 4;
const SHAPE_TYPE_POLYLINE: u32 = 5;
const SHAPE_TYPE_TRIMESH: u32 = 6;
const SHAPE_TYPE_CONVEX_POLY: u32 = 7;
// TODO: since this shape type is only for trimesh, it doesn’t implement all the
//       operations it could if it were a standalone shape.
const SHAPE_TYPE_TRIANGLE: u32 = 8;

/// A generic shape that can represent any concrete shape type.
///
/// This is a tagged union encoded in two vec4 values. The shape type
/// is stored in the 'a.w' component as a bitcast u32.
struct Shape {
    /// First vec4 containing shape-specific data and type tag in 'w' component.
    a: vec4<f32>,
    /// Second vec4 for additional shape data (primarily for capsule segment endpoint).
    b: vec4<f32>,
    /// Third vec4, only used for triangles.
    c: vec4<f32>,
}

fn shape_type(shape: Shape) -> u32 {
    return bitcast<u32>(shape.a.w);
}

struct PfmSubShape {
    shape: Shape,
    thickness: f32,
    valid: bool,
}

/*
 *
 * Shape conversions.
 *
 */

fn to_ball(shape: Shape) -> Bal::Ball {
    // Ball layout:
    //     vec4(radius, _, _, shape_type)
    //     vec4(_, _, _, _)
    return Bal::Ball(shape.a.x);
}

fn to_triangle(shape: Shape) -> Tri::Triangle {
    // Triangle layout:
    //     vec4(a.x, a.y, a.z, shape_type)
    //     vec4(b.x, b.y, b.z, _)
    //     vec4(c.x, c.y, c.z, _)
    #if DIM == 2
    return Tri::Triangle(shape.a.xy, shape.b.xy, shape.c.xy);
    #else
    return Tri::Triangle(shape.a.xyz, shape.b.xyz, shape.c.xyz);
    #endif
}

fn from_triangle(tri: Tri::Triangle) -> Shape {
    let tag = bitcast<f32>(SHAPE_TYPE_TRIANGLE);
    #if DIM == 2
    return Shape(
        vec4(tri.a, 0.0, tag),
        vec4(tri.b, 0.0, 0.0),
        vec4(tri.c, 0.0, 0.0)
    );
    #else
    return Shape(
        vec4(tri.a, tag),
        vec4(tri.b, 0.0),
        vec4(tri.c, 0.0)
    );
    #endif
}

fn to_capsule(shape: Shape) -> Cap::Capsule {
    // Capsule layout:
    //     vec4(ax, ay, az, shape_type)
    //     vec4(bx, by, bz, radius)
#if DIM == 2
    return Cap::Capsule(Seg::Segment(shape.a.xy, shape.b.xy), shape.b.w);
#else
    return Cap::Capsule(Seg::Segment(shape.a.xyz, shape.b.xyz), shape.b.w);
#endif
}

fn from_capsule(cap: Cap::Capsule) -> Shape {
#if DIM == 2
    let a = vec4(cap.segment.a, 0.0, bitcast<f32>(SHAPE_TYPE_CAPSULE));
    let b = vec4(cap.segment.b, 0.0, cap.radius);
#else
    let a = vec4(cap.segment.a, bitcast<f32>(SHAPE_TYPE_CAPSULE));
    let b = vec4(cap.segment.b, cap.radius);
#endif
    return Shape(a, b, vec4());
}

fn to_cuboid(shape: Shape) -> Cub::Cuboid {
    // Cuboid layout:
    //     vec4(hx, hy, hz, shape_type)
    //     vec4(_, _, _, _)
#if DIM == 2
    return Cub::Cuboid(shape.a.xy);
#else
    return Cub::Cuboid(shape.a.xyz);
#endif
}

#if DIM == 3
    fn to_cone(shape: Shape) -> Con::Cone {
        // Cone layout:
        //     vec4(half_height, radius, _, shape_type)
        //     vec4(_, _, _, _)
        return Con::Cone(shape.a.x, shape.a.y);
    }

    fn to_cylinder(shape: Shape) -> Cyl::Cylinder {
        // Cylinder layout:
        //     vec4(half_height, radius, _, shape_type)
        //     vec4(_, _, _, _)
        return Cyl::Cylinder(shape.a.x, shape.a.y);
    }
#endif

fn to_convex_poly(shape: Shape) -> Convex::ConvexPolyhedron {
    // Convex polyhedron layout:
    //     vec4(first_vtx_id, end_vtx_id, _, shape_type)
    //     vec4(first_tri_id, end_tri_id, _, _)
    let first_vtx_id = bitcast<u32>(shape.a.x);
    let end_vtx_id = bitcast<u32>(shape.a.y);
    let first_tri_id = bitcast<u32>(shape.b.x);
    let end_tri_id = bitcast<u32>(shape.b.y);
    return Convex::ConvexPolyhedron(first_vtx_id, end_vtx_id, first_tri_id, end_tri_id);
}

fn to_trimesh(shape: Shape) -> TriMesh::TriMesh {
    // Trimesh layout:
    //     vec4(bvh_vtx_root_id, bvh_idx_root_id, first_tri_id, shape_type)
    //     vec4(root_aabb.mins, _)
    //     vec4(root_aabb.maxs, _)
    let bvh_vtx_root_id = bitcast<u32>(shape.a.x);
    let bvh_idx_root_id = bitcast<u32>(shape.a.y);
    let first_tri_id = bitcast<u32>(shape.a.z);
    #if DIM == 2
    let root_aabb = Aabb::Aabb(shape.b.xy, shape.c.xy);
    #else
    let root_aabb = Aabb::Aabb(shape.b.xyz, shape.c.xyz);
    #endif
    return TriMesh::TriMesh(bvh_vtx_root_id, bvh_idx_root_id, first_tri_id, root_aabb);
}

fn to_polyline(shape: Shape) -> Polyline::Polyline {
    // Polyline layout:
    //     vec4(bvh_vtx_root_id, bvh_idx_root_id, first_seg_id, shape_type)
    //     vec4(root_aabb.mins, _)
    //     vec4(root_aabb.maxs, _)
    let bvh_vtx_root_id = bitcast<u32>(shape.a.x);
    let bvh_idx_root_id = bitcast<u32>(shape.a.y);
    let first_seg_id = bitcast<u32>(shape.a.z);
    #if DIM == 2
    let root_aabb = Aabb::Aabb(shape.b.xy, shape.c.xy);
    #else
    let root_aabb = Aabb::Aabb(shape.b.xyz, shape.c.xyz);
    #endif
    return Polyline::Polyline(bvh_vtx_root_id, bvh_idx_root_id, first_seg_id, root_aabb);
}

/*
 *
 * Geometric operations.
 *
 */
/// Projects a point on this shape.
///
/// If the point is inside the shape, the point itself is returned.
fn projectLocalPoint(shape: Shape, pt: Vector) -> Vector {
    let ty = shape_type(shape);
    if ty == SHAPE_TYPE_BALL {
        return Bal::projectLocalPoint(to_ball(shape), pt);
    }
    if ty == SHAPE_TYPE_CUBOID {
        return Cub::projectLocalPoint(to_cuboid(shape), pt);
    }
    if ty == SHAPE_TYPE_CAPSULE {
        return Cap::projectLocalPoint(to_capsule(shape), pt);
    }
#if DIM == 3
    if ty == SHAPE_TYPE_CONE {
        return Con::projectLocalPoint(to_cone(shape), pt);
    }
    if ty == SHAPE_TYPE_CYLINDER {
        return Cyl::projectLocalPoint(to_cylinder(shape), pt);
    }
#endif
    return pt;
}

/// Projects a point on a transformed shape.
///
/// If the point is inside the shape, the point itself is returned.
fn projectPoint(shape: Shape, pose: Transform, pt: Vector) -> Vector {
    let ty = shape_type(shape);
    if ty == SHAPE_TYPE_BALL {
        return Bal::projectPoint(to_ball(shape), pose, pt);
    }
    if ty == SHAPE_TYPE_CUBOID {
        return Cub::projectPoint(to_cuboid(shape), pose, pt);
    }
    if ty == SHAPE_TYPE_CAPSULE {
        return Cap::projectPoint(to_capsule(shape), pose, pt);
    }
#if DIM == 3
    if ty == SHAPE_TYPE_CONE {
        return Con::projectPoint(to_cone(shape), pose, pt);
    }
    if ty == SHAPE_TYPE_CYLINDER {
        return Cyl::projectPoint(to_cylinder(shape), pose, pt);
    }
#endif
    return pt;
}


/// Projects a point on the boundary of a shape.
fn projectLocalPointOnBoundary(shape: Shape, pt: Vector) -> Proj::ProjectionResult {
    let ty = shape_type(shape);
    if ty == SHAPE_TYPE_BALL {
        return Bal::projectLocalPointOnBoundary(to_ball(shape), pt);
    }
    if ty == SHAPE_TYPE_CUBOID {
        return Cub::projectLocalPointOnBoundary(to_cuboid(shape), pt);
    }
    if ty == SHAPE_TYPE_CAPSULE {
        return Cap::projectLocalPointOnBoundary(to_capsule(shape), pt);
    }
#if DIM == 3
    if ty == SHAPE_TYPE_CONE {
        return Con::projectLocalPointOnBoundary(to_cone(shape), pt);
    }
    if ty == SHAPE_TYPE_CYLINDER {
        return Cyl::projectLocalPointOnBoundary(to_cylinder(shape), pt);
    }
#endif
    return Proj::ProjectionResult(pt, false);
}

/// Project a point of a transformed shape’s boundary.
///
/// If the point is inside of the shape, it will be projected on its boundary but
/// `ProjectionResult::is_inside` will be set to `true`.
fn projectPointOnBoundary(shape: Shape, pose: Transform, pt: Vector) -> Proj::ProjectionResult {
    let ty = shape_type(shape);
    if ty == SHAPE_TYPE_BALL {
        return Bal::projectPointOnBoundary(to_ball(shape), pose, pt);
    }
    if ty == SHAPE_TYPE_CUBOID {
        return Cub::projectPointOnBoundary(to_cuboid(shape), pose, pt);
    }
    if ty == SHAPE_TYPE_CAPSULE {
        return Cap::projectPointOnBoundary(to_capsule(shape), pose, pt);
    }
#if DIM == 3
    if ty == SHAPE_TYPE_CONE {
        return Con::projectPointOnBoundary(to_cone(shape), pose, pt);
    }
    if ty == SHAPE_TYPE_CYLINDER {
        return Cyl::projectPointOnBoundary(to_cylinder(shape), pose, pt);
    }
#endif
    return Proj::ProjectionResult(pt, false);
}

fn support_point(shape: Shape, pose: Transform, axis: Vector) -> Vector {
    let local_axis = Pose::invMulVec(pose, axis);
    let local_pt = local_support_point(shape, local_axis);
    return Pose::mulPt(pose, local_pt);
}

fn local_support_point(shape: Shape, dir: Vector) -> Vector {
    let ty = shape_type(shape);
//    if ty == SHAPE_TYPE_BALL {
//        return Bal::local_support_point(to_ball(shape), dir);
//    }
    if ty == SHAPE_TYPE_CUBOID {
        return Cub::local_support_point(to_cuboid(shape), dir);
    }
    if ty == SHAPE_TYPE_TRIANGLE {
        return Tri::local_support_point(to_triangle(shape), dir);
    }
    if ty == SHAPE_TYPE_CAPSULE {
        return Cap::local_support_point(to_capsule(shape), dir);
    }
#if DIM == 3
    if ty == SHAPE_TYPE_CONE {
        return Con::local_support_point(to_cone(shape), dir);
    }
    if ty == SHAPE_TYPE_CYLINDER {
        return Cyl::local_support_point(to_cylinder(shape), dir);
    }
#endif

    if ty == SHAPE_TYPE_CONVEX_POLY {
        return Convex::local_support_point(to_convex_poly(shape), dir);
    }

    return Vector();
}

fn support_face(shape: Shape, dir: Vector) -> Feat::PolygonalFeature {
    let ty = shape_type(shape);
//    if ty == SHAPE_TYPE_BALL {
//        return Bal::support_face(to_ball(shape), dir);
//    }
    if ty == SHAPE_TYPE_CUBOID {
        return Cub::support_face(to_cuboid(shape), dir);
    }
    if ty == SHAPE_TYPE_TRIANGLE {
        return Tri::support_face(to_triangle(shape), dir);
    }
    if ty == SHAPE_TYPE_CAPSULE {
        return Cap::support_face(to_capsule(shape), dir);
    }
#if DIM == 3
    if ty == SHAPE_TYPE_CONE {
        return Con::support_face(to_cone(shape), dir);
    }
    if ty == SHAPE_TYPE_CYLINDER {
        return Cyl::support_face(to_cylinder(shape), dir);
    }
#endif

    if ty == SHAPE_TYPE_CONVEX_POLY {
        return Convex::support_face(to_convex_poly(shape), dir);
    }

    return Feat::PolygonalFeature();
}

fn pfm_subshape(shape: Shape) -> PfmSubShape {
    let ty = shape_type(shape);
    if ty == SHAPE_TYPE_CUBOID || ty == SHAPE_TYPE_CONE || ty == SHAPE_TYPE_CYLINDER || ty == SHAPE_TYPE_CONVEX_POLY || ty == SHAPE_TYPE_TRIANGLE {
        // No subshape, return the original shape itself.
        return PfmSubShape(shape, 0.0, true);
    }

    if ty == SHAPE_TYPE_BALL {
        let ball = to_ball(shape);
        let segment = Cap::Capsule();
        return PfmSubShape(from_capsule(segment), ball.radius, true);
    }

    if ty == SHAPE_TYPE_CAPSULE {
        let capsule = to_capsule(shape);
        let without_radius = Cap::Capsule(capsule.segment, 0.0);
        return PfmSubShape(from_capsule(without_radius), capsule.radius, true);
    }

    // Not a PFM.
    return PfmSubShape(shape, 0.0, false);
}


/// Creates an AABB from a transformed shape.
fn aabb(pose: Transform, shape: Shape) -> Aabb::Aabb {
    let ty = shape_type(shape);
    if ty == SHAPE_TYPE_BALL {
        let ball = to_ball(shape);
#if DIM == 2
        let tra = pose.translation;
        let rad = ball.radius * pose.scale;
#else
        let tra = pose.translation_scale.xyz;
        let rad = ball.radius * pose.translation_scale.w;
#endif

        return Aabb::Aabb(
            tra - Vector(rad),
            tra + Vector(rad)
        );
    }

    if ty == SHAPE_TYPE_CUBOID {
        let cuboid = to_cuboid(shape);
        let local_aabb = Aabb::Aabb(-cuboid.halfExtents, cuboid.halfExtents);
        return Aabb::transform(local_aabb, pose);
    }

    if ty == SHAPE_TYPE_TRIANGLE {
        let triangle = to_triangle(shape);
        let local_aabb = Tri::aabb(triangle);
        return Aabb::transform(local_aabb, pose);
    }

    if ty == SHAPE_TYPE_CAPSULE {
        let capsule = to_capsule(shape);
        let aa = Pose::mulPt(pose, capsule.segment.a);
        let bb = Pose::mulPt(pose, capsule.segment.b);
        return Aabb::Aabb(
            min(aa, bb) - Vector(capsule.radius),
            max(aa, bb) + Vector(capsule.radius),
        );
    }

#if DIM == 3
    if ty == SHAPE_TYPE_CONE {
        let cone = to_cone(shape);
        let local_aabb = Aabb::Aabb(
            -vec3(cone.radius, cone.half_height, cone.radius),
            vec3(cone.radius, cone.half_height, cone.radius),
        );
        return Aabb::transform(local_aabb, pose);
    }

    if ty == SHAPE_TYPE_CYLINDER {
        let cylinder = to_cylinder(shape);
        let local_aabb = Aabb::Aabb(
            -vec3(cylinder.radius, cylinder.half_height, cylinder.radius),
            vec3(cylinder.radius, cylinder.half_height, cylinder.radius),
        );
        return Aabb::transform(local_aabb, pose);
    }
#endif

    if ty == SHAPE_TYPE_CONVEX_POLY {
        let poly = to_convex_poly(shape);
        let local_aabb = Convex::aabb(poly);
        return Aabb::transform(local_aabb, pose);
    }

    if ty == SHAPE_TYPE_TRIMESH {
        let trimesh = to_trimesh(shape);
        let local_aabb = TriMesh::aabb(trimesh);
        return Aabb::transform(local_aabb, pose);
    }

    if ty == SHAPE_TYPE_POLYLINE {
        let polyline = to_polyline(shape);
        let local_aabb = Polyline::aabb(polyline);
        return Aabb::transform(local_aabb, pose);
    }

    return Aabb::Aabb();
}