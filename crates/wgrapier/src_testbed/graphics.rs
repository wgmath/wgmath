#[cfg(feature = "dim2")]
use rapier2d as rapier;
#[cfg(feature = "dim3")]
use rapier3d as rapier;

use crate::backend::PhysicsBackend;
use kiss3d::nalgebra::Point3;
use kiss3d::window::Window;
use rapier::math::DIM;
use rapier::parry::shape::ShapeType;
use std::collections::HashMap;
use std::ops::MulAssign;

use crate::SimulationState;
#[cfg(feature = "dim3")]
use kiss3d::prelude::InstanceData;
#[cfg(feature = "dim2")]
use kiss3d::prelude::{PlanarInstanceData, PlanarSceneNode};
#[cfg(feature = "dim3")]
use kiss3d::scene::SceneNode;
#[cfg(feature = "dim3")]
use nalgebra::Vector3;

pub struct InstancedNodeEntry {
    pub index: usize,
    pub color: [f32; 4],
    pub scale: [f32; DIM],
}

pub struct InstancedNode {
    #[cfg(feature = "dim2")]
    pub node: PlanarSceneNode,
    #[cfg(feature = "dim3")]
    pub node: SceneNode,
    pub entries: Vec<InstancedNodeEntry>,
    #[cfg(feature = "dim2")]
    pub data: Vec<PlanarInstanceData>,
    #[cfg(feature = "dim3")]
    pub data: Vec<InstanceData>,
}

pub struct RenderContext {
    pub instances: Vec<InstancedNode>,
}

impl RenderContext {
    pub fn clear(&mut self) {
        for instance in &mut self.instances {
            instance.node.unlink();
        }
        self.instances.clear();
    }
}

/// Set up a simple scene using instancing for efficient rendering
pub async fn setup_graphics(window: &mut Window, phys: &SimulationState) -> RenderContext {
    let colors = [
        Point3::new(124.0 / 255.0, 144.0 / 255.0, 1.0),
        Point3::new(8.0 / 255.0, 144.0 / 255.0, 1.0),
        Point3::new(124.0 / 255.0, 7.0 / 255.0, 1.0),
        Point3::new(124.0 / 255.0, 144.0 / 255.0, 7.0 / 255.0),
        Point3::new(200.0 / 255.0, 37.0 / 255.0, 1.0),
        Point3::new(124.0 / 255.0, 230.0 / 255.0, 25.0 / 255.0),
    ];
    let fixed_color = Point3::new(0.6, 0.6, 0.6);

    let mut instances = HashMap::new();

    for (i, (_, co)) in phys.colliders.iter().enumerate() {
        let shape = co.shape();
        let is_fixed = co.parent().map(|h| phys.bodies[h].is_fixed()) != Some(false);
        let color = if is_fixed {
            fixed_color
        } else {
            colors[i % colors.len()]
        };

        match shape.shape_type() {
            ShapeType::Ball => {
                let instanced_node = instances.entry(ShapeType::Ball).or_insert_with(|| {
                    #[cfg(feature = "dim2")]
                    let node = window.add_circle(0.5);
                    #[cfg(feature = "dim3")]
                    let node = {
                        // NOTE: the default kiss3d sphere is a bit slow to render when we have
                        //       100K+ instances because it uses a lot of subdivision. Create one
                        //       with lower details.
                        let lowres_sphere = kiss3d::procedural::sphere(1.0, 10, 10, true);
                        window.add_render_mesh(lowres_sphere, Vector3::repeat(1.0))
                    };
                    InstancedNode {
                        node,
                        entries: vec![],
                        data: vec![],
                    }
                });
                let ball = shape.as_ball().unwrap();
                instanced_node.entries.push(InstancedNodeEntry {
                    index: i,
                    color: [
                        (color.x + 0.4) % 1.0, // Shift the color palette so that balls are colored differently than cubes
                        (color.y + 0.4) % 1.0, // Shift the color palette so that balls are colored differently than cubes
                        (color.z + 0.4) % 1.0, // Shift the color palette so that balls are colored differently than cubes
                        1.0,
                    ],
                    scale: [ball.radius * 2.0; DIM],
                })
            }
            ShapeType::Cuboid => {
                let instanced_node = instances.entry(ShapeType::Cuboid).or_insert_with(|| {
                    #[cfg(feature = "dim2")]
                    let node = window.add_rectangle(1.0, 1.0);
                    #[cfg(feature = "dim3")]
                    let node = window.add_cube(1.0, 1.0, 1.0);
                    InstancedNode {
                        node,
                        entries: vec![],
                        data: vec![],
                    }
                });
                let cuboid = shape.as_cuboid().unwrap();
                instanced_node.entries.push(InstancedNodeEntry {
                    index: i,
                    color: [color.x, color.y, color.z, 1.0],
                    scale: (cuboid.half_extents * 2.0).into(),
                })
            }
            ShapeType::Cylinder => {
                let instanced_node = instances.entry(ShapeType::Cylinder).or_insert_with(|| {
                    #[cfg(feature = "dim2")]
                    let node = window.add_rectangle(1.0, 1.0);
                    #[cfg(feature = "dim3")]
                    let node = window.add_cylinder(1.0, 1.0);
                    InstancedNode {
                        node,
                        entries: vec![],
                        data: vec![],
                    }
                });
                let cyl = shape.as_cylinder().unwrap();
                instanced_node.entries.push(InstancedNodeEntry {
                    index: i,
                    color: [color.x, color.y, color.z, 1.0],
                    scale: [cyl.radius, cyl.half_height * 2.0, cyl.radius],
                })
            }
            ShapeType::Cone => {
                let instanced_node = instances.entry(ShapeType::Cone).or_insert_with(|| {
                    #[cfg(feature = "dim2")]
                    let node = window.add_rectangle(1.0, 1.0);
                    #[cfg(feature = "dim3")]
                    let node = window.add_cone(1.0, 1.0);
                    InstancedNode {
                        node,
                        entries: vec![],
                        data: vec![],
                    }
                });
                let cyl = shape.as_cone().unwrap();
                instanced_node.entries.push(InstancedNodeEntry {
                    index: i,
                    color: [color.x, color.y, color.z, 1.0],
                    scale: [cyl.radius, cyl.half_height * 2.0, cyl.radius],
                })
            }
            _ => todo!(),
        }
    }

    RenderContext {
        instances: instances.into_values().collect(),
    }
}

/// Update rendering instances with current physics poses
pub fn update_instances(render_ctx: &mut RenderContext, physics_backend: &PhysicsBackend) {
    for instanced_node in &mut render_ctx.instances {
        instanced_node.data.clear();

        for entry in &instanced_node.entries {
            let pose = physics_backend.poses()[entry.index];

            #[cfg(feature = "dim2")]
            {
                let position = pose.similarity.isometry.translation.vector.into();
                // Convert rotation to a 2x2 matrix
                let mut deformation = pose
                    .similarity
                    .isometry
                    .rotation
                    .to_rotation_matrix()
                    .into_inner();
                deformation.column_mut(0).mul_assign(entry.scale[0]);
                deformation.column_mut(1).mul_assign(entry.scale[1]);

                instanced_node.data.push(PlanarInstanceData {
                    position,
                    deformation,
                    color: entry.color,
                });
            }

            #[cfg(feature = "dim3")]
            {
                let position = pose.isometry.translation.vector.into();

                // Convert rotation to a 3x3 matrix
                let mut deformation = pose.isometry.rotation.to_rotation_matrix().into_inner();
                deformation.column_mut(0).mul_assign(entry.scale[0]);
                deformation.column_mut(1).mul_assign(entry.scale[1]);
                deformation.column_mut(2).mul_assign(entry.scale[2]);

                instanced_node.data.push(InstanceData {
                    position,
                    deformation,
                    color: entry.color,
                });
            }
        }

        instanced_node.node.set_instances(&instanced_node.data);
    }
}
