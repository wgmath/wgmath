#define_import_path wgparry::bounding_volumes::aabb

#import wgebra::sim3 as Pose;
#import wgparry::shape as Shape;

struct Aabb {
    mins: Vector,
    maxs: Vector
}

fn from_shape(pose: Pose::Sim3, shape: Shape::Shape) -> Aabb {
    let ty = Shape::shape_type(shape);
    if ty == Shape::SHAPE_TYPE_BALL {
        let ball = Shape::to_ball(shape);
        let tra = pose.translation_scale.xyz;
        let rad = ball.radius * pose.translation_scale.w;
        return Aabb(
            tra - vec3(rad),
            tra + vec3(rad)
        );
    }

    if ty == Shape::SHAPE_TYPE_CUBOID {
        let cuboid = Shape::to_cuboid(shape);
        // FIXME
        return Aabb();
    }

    if ty == Shape::SHAPE_TYPE_CAPSULE {
        let capsule = Shape::to_capsule(shape);
        // FIXME
        return Aabb();
    }

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

    // TODO: not implemented.
    return Aabb();
}

fn check_intersection(aabb1: Aabb, aabb2: Aabb) -> bool {
    return !(any(aabb2.maxs < aabb1.mins) || any(aabb1.maxs < aabb2.mins));
}

fn merge(aabb1: Aabb, aabb2: Aabb) -> Aabb {
    return Aabb(min(aabb1.mins, aabb2.mins), max(aabb1.maxs, aabb2.maxs));
}