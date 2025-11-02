use rapier3d::prelude::*;
use wgrapier_testbed3d::SimulationState;

fn create_pyramid(
    bodies: &mut RigidBodySet,
    colliders: &mut ColliderSet,
    offset: Vector<f32>,
    stack_height: usize,
    rad: f32,
) {
    let shift = rad * 2.0;

    for i in 0usize..stack_height {
        for j in i..stack_height {
            let fj = j as f32;
            let fi = i as f32;
            let x = (fi * shift / 2.0) + (fj - fi) * shift;
            let y = fi * shift;

            // Build the rigid body.
            let rigid_body = RigidBodyBuilder::dynamic().translation(vector![x, y, 0.0] + offset);
            let handle = bodies.insert(rigid_body);
            let collider = ColliderBuilder::cuboid(rad, rad, rad);
            colliders.insert_with_parent(collider, handle, bodies);
        }
    }
}

pub fn init_world() -> SimulationState {
    /*
     * World
     */
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let impulse_joints = ImpulseJointSet::new();
    let _multibody_joints = MultibodyJointSet::new();

    let rad = 0.5;
    let pyramid_count = 40;
    let spacing = 4.0;

    /*
     * Ground
     */
    let ground_size = 100.0;
    let ground_height = 0.1;

    let rigid_body = RigidBodyBuilder::fixed().translation(vector![0.0, -ground_height, 0.0]);
    let ground_handle = bodies.insert(rigid_body);
    let collider = ColliderBuilder::cuboid(
        ground_size,
        ground_height,
        pyramid_count as f32 * spacing / 2.0 + ground_size,
    );
    colliders.insert_with_parent(collider, ground_handle, &mut bodies);

    /*
     * Create the cubes
     */
    for pyramid_index in 0..pyramid_count {
        let bottomy = rad;
        create_pyramid(
            &mut bodies,
            &mut colliders,
            vector![
                0.0,
                bottomy,
                (pyramid_index as f32 - pyramid_count as f32 / 2.0) * spacing
            ],
            60,
            rad,
        );

        create_pyramid(
            &mut bodies,
            &mut colliders,
            vector![
                -75.0,
                bottomy,
                (pyramid_index as f32 - pyramid_count as f32 / 2.0) * spacing
            ],
            60,
            rad,
        );
    }

    /*
     * Set up the testbed.
     */
    SimulationState {
        bodies,
        colliders,
        impulse_joints,
    }
    // testbed.look_at(point![100.0, 100.0, 100.0], Point::origin());
}
