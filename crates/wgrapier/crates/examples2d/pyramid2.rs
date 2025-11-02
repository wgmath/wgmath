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
    let ground_size = 500.0;
    let ground_thickness = 1.0;

    let rigid_body = RigidBodyBuilder::fixed();
    let ground_handle = bodies.insert(rigid_body);
    let collider = ColliderBuilder::cuboid(ground_size, ground_thickness);
    colliders.insert_with_parent(collider, ground_handle, &mut bodies);

    /*
     * Create the cubes
     */
    let num = 200;
    let rad = 0.5;

    let shiftx = rad * 2.0 + 0.1;
    let shifty = rad * 2.0;
    let centerx = shiftx * (num as f32) / 2.0;
    let centery = shifty / 2.0 + ground_thickness;

    for k in 0..4 {
        for i in 0usize..num {
            for j in i..num {
                let fj = j as f32;
                let fi = i as f32;
                let x = (fi * shiftx / 2.0) + (fj - fi) * shiftx - centerx
                    + (k as f32 - 1.5) * rad * 2.5 * num as f32;
                let y = fi * shifty + centery;

                // Build the rigid body.
                let rigid_body = RigidBodyBuilder::dynamic().translation(vector![x, y]);
                let handle = bodies.insert(rigid_body);
                let collider = ColliderBuilder::cuboid(rad, rad);
                colliders.insert_with_parent(collider, handle, &mut bodies);
            }
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
    // testbed.look_at(point![0.0, 2.5], 5.0);
}
