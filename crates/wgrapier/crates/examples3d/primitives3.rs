use rapier3d::na::Vector3;
use rapier3d::prelude::*;
use wgrapier_testbed3d::SimulationState;

pub fn init_world() -> SimulationState {
    const NXZ: isize = 20;
    const NY: isize = 40;

    let mut bodies = RigidBodySet::default();
    let mut colliders = ColliderSet::default();
    let impulse_joints = ImpulseJointSet::default();

    /*
     * Falling dynamic objects.
     */
    for j in 0..NY {
        let max_ik = NXZ / 2;
        for i in -max_ik..max_ik {
            for k in -max_ik..max_ik {
                let x = i as f32 * 1.1 + j as f32 * 0.01;
                let y = j as f32 * 1.1 + 1.0;
                let z = k as f32 * 1.1 + j as f32 * 0.01;
                let pos = Vector3::new(x, y, z);
                let body = bodies.insert(RigidBodyBuilder::dynamic().translation(pos));

                let collider = match j % 3 {
                    0 => ColliderBuilder::cylinder(0.5, 0.5),
                    1 => ColliderBuilder::cuboid(0.5, 0.5, 0.5),
                    _ => ColliderBuilder::cone(0.5, 0.5),
                };

                colliders.insert_with_parent(collider, body, &mut bodies);
            }
        }
    }

    /*
     * Floor made of large cuboids.
     */
    {
        let thick = NXZ as f32 * 1.3;
        let height = 8.0;
        let walls = [
            (
                Vector3::new(0.0, -0.5, 0.0),
                Vector3::new(thick, 0.5, thick),
            ),
            (
                Vector3::new(thick, height, 0.0),
                Vector3::new(0.5, height, thick),
            ),
            (
                Vector3::new(-thick, height, 0.0),
                Vector3::new(0.5, height, thick),
            ),
            (
                Vector3::new(0.0, height, thick),
                Vector3::new(thick, height, 0.5),
            ),
            (
                Vector3::new(0.0, height, -thick),
                Vector3::new(thick, height, 0.5),
            ),
        ];

        for (wall_pos, wall_sz) in walls {
            colliders.insert(
                ColliderBuilder::cuboid(wall_sz.x, wall_sz.y, wall_sz.z).translation(wall_pos),
            );
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
    // testbed.look_at(point![100.0, 100.0, 100.0], Point::origin());
}
