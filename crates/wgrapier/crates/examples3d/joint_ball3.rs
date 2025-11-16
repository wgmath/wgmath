use rapier3d::prelude::*;
use wgrapier_testbed3d::SimulationState;

pub fn init_world() -> SimulationState {
    /*
     * World
     */
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let mut impulse_joints = ImpulseJointSet::new();

    let rad = 0.4;
    let ni = 200;
    let nk = 301;
    let shift = 1.0;
    let center = vector![nk as f32 * shift / 2.0, 0.0, ni as f32 * shift / 2.0];

    let mut body_handles = Vec::new();

    // A lot of joints. Kind of like a piece of cloth.
    for k in 0..nk {
        for i in 0..ni {
            let fk = k as f32;
            let fi = i as f32;

            let status = if ((i == 0 || i == ni - 1) && (k % 4 == 0 || k == ni - 1))
                || ((k == 0 || k == nk - 1) && (i % 4 == 0 || i == nk - 1))
            {
                RigidBodyType::Fixed
            } else {
                RigidBodyType::Dynamic
            };

            let rigid_body = RigidBodyBuilder::new(status)
                .translation(vector![fk * shift, 0.0, fi * shift] - center);
            let child_handle = bodies.insert(rigid_body);
            let collider = if status == RigidBodyType::Fixed {
                ColliderBuilder::cuboid(rad, rad, rad)
            } else {
                ColliderBuilder::ball(rad)
            };
            colliders.insert_with_parent(collider, child_handle, &mut bodies);

            // Vertical joint.
            if i > 0 {
                let parent_handle = *body_handles.last().unwrap();
                let joint = SphericalJointBuilder::new().local_anchor2(point![0.0, 0.0, -shift]);
                impulse_joints.insert(parent_handle, child_handle, joint, true);
            }

            // Horizontal joint.
            if k > 0 {
                let parent_index = body_handles.len() - ni;
                let parent_handle = body_handles[parent_index];
                let joint = SphericalJointBuilder::new().local_anchor2(point![-shift, 0.0, 0.0]);
                impulse_joints.insert(parent_handle, child_handle, joint, true);
            }

            body_handles.push(child_handle);
        }
    }

    // Some rigid-bodies to fall on top.
    let nj = 10;
    let nk = nk / 3;
    let ni = ni / 6;
    let rad = rad * 2.5;

    for k in 0..nk {
        for i in 0..ni {
            for j in 0..nj {
                let body = RigidBodyBuilder::dynamic().translation(vector![
                    (k as f32 - nk as f32 / 2.0) * rad * 2.1,
                    j as f32 * rad * 2.1 + 2.0,
                    (i as f32 - ni as f32 / 2.0) * rad * 2.1,
                ]);
                let handle = bodies.insert(body);
                let collider = ColliderBuilder::cuboid(rad, rad, rad);
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
    // testbed.look_at(point![-110.0, -46.0, 170.0], point![54.0, -38.0, 29.0]);
}
