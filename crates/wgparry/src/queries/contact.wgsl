#import wgebra::sim3 as Pose
#import wgparry::ball as Ball

#define_import_path wgparry::contact

/// A pair of contact points between two shapes.
struct Contact {
    /// The contact point on the first shape.
    point1: vec3<f32>,
    /// The contact pointon the second shape.
    point2: vec3<f32>,
    /// The first shape’s normal at its contact point.
    normal1: vec3<f32>,
    /// The second shape’s normal at its contact point.
    normal2: vec3<f32>,
    /// The distance between the two contact points.
    dist: f32,
}


/// Computes the contact between two balls.
fn ballBall(pose12: Pose::Sim3, ball1: Ball::Ball, ball2: Ball::Ball) -> Contact {
    let r1 = ball1.radius;
    let r2 = ball2.radius;
    let center2_1 = pose12.translation_scale.xyz;
    let distance = length(center2_1);
    let sum_radius = r1 + r2;

    var normal1 = vec3(1.0, 0.0, 0.0);

    if distance != 0.0 {
        normal1 = center2_1 / distance;
    }

    let normal2 = -Pose::invMulUnitVec(pose12, normal1);
    let point1 = normal1 * r1;
    let point2 = normal2 * r2;

    return Contact(
        point1,
        point2,
        normal1,
        normal2,
        distance - sum_radius,
    );
}