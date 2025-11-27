use rapier2d::prelude::*;
use wgrapier_testbed2d::SimulationState;

pub fn init_world() -> SimulationState {
    /*
     * World
     */
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let impulse_joints = ImpulseJointSet::new();
    let _multibody_joints = MultibodyJointSet::new();

    /*
     * Ground
     */
    let ground_size = 200.0;
    let nsubdivs = 40;
    let step_size = ground_size / (nsubdivs as f32);
    let mut points = Vec::new();

    points.push(point![-ground_size / 2.0, 240.0]);
    for i in 1..nsubdivs - 1 {
        let x = -ground_size / 2.0 + i as f32 * step_size;
        let y = (i as f32 / nsubdivs as f32 * 10.0).cos() * 20.0;
        points.push(point![x, y]);
    }
    points.push(point![ground_size / 2.0, 240.0]);

    let rigid_body = RigidBodyBuilder::fixed();
    let handle = bodies.insert(rigid_body);
    let collider = ColliderBuilder::polyline(points, None);
    colliders.insert_with_parent(collider, handle, &mut bodies);

    /*
     * Create the cubes
     */
    let num = 100;
    let rad = 0.5;

    let shift = rad * 2.0 + 0.4;
    let centerx = shift * (num as f32) / 2.0;
    let centery = shift / 2.0;
    let mut rng = oorandom::Rand32::new(0);

    for i in 0..num {
        for j in 0usize..num * 3 {
            let x = i as f32 * shift - centerx + (j % 2) as f32 * 0.2;
            let y = j as f32 * shift + centery + 20.0;

            // Build the rigid body.
            let rigid_body = RigidBodyBuilder::dynamic().translation(vector![x, y]);
            let handle = bodies.insert(rigid_body);
            let collider = match j % 4 {
                0 => ColliderBuilder::cuboid(rad, rad),
                1 => ColliderBuilder::capsule_y(rad, rad),
                2 => {
                    if i % 2 == 0 {
                        continue;
                    }

                    // Make a convex polygon.
                    let mut points = Vec::new();
                    let scale = 2.0;
                    for _ in 0..10 {
                        let pt = Point::new(rng.rand_float() - 0.5, rng.rand_float() - 0.5);
                        points.push(pt * scale);
                    }

                    // TODO: align the collider’s local origin to its center-of-mass.
                    //       wgrapier currently doesn’t support misaligned center-of-masses.
                    let shape = SharedShape::convex_hull(&points).unwrap();
                    let mprops = shape.mass_properties(1.0);
                    points
                        .iter_mut()
                        .for_each(|pt| *pt -= mprops.local_com.coords);

                    ColliderBuilder::convex_hull(&points).unwrap()
                }
                _ => ColliderBuilder::ball(rad),
            };
            colliders.insert_with_parent(collider, handle, &mut bodies);
        }
    }

    /*
     * Set up the testbed.
     */
    SimulationState {
        bodies,
        colliders,
        impulse_joints,
    }
    // testbed.look_at(point![0.0, 50.0], 10.0);
}
