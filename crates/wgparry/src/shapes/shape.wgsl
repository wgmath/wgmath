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

#define_import_path wgparry::shape

const SHAPE_TYPE_BALL: u32 = 0;
const SHAPE_TYPE_CUBOID: u32 = 1;
const SHAPE_TYPE_CAPSULE: u32 = 2;
const SHAPE_TYPE_CONE: u32 = 3;
const SHAPE_TYPE_CYLINDER: u32 = 4;
const SHAPE_TYPE_POLYLINE: u32 = 5;
const SHAPE_TYPE_TRIMESH: u32 = 6;

// A generic shape. This is an enum that might contain any shape information.
// PERF: if it wasn’t for the capsule, we could store all the shapes into a single vec4…
struct Shape {
    a: vec4<f32>,
    b: vec4<f32>,
}

fn shape_type(shape: Shape) -> u32 {
    return bitcast<u32>(shape.a.w);
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
