#[cfg(feature = "dim2")]
use rapier2d as rapier;
#[cfg(feature = "dim3")]
use rapier3d as rapier;

use crate::backend::PhysicsBackend;
use crate::SimulationState;
use kiss3d::nalgebra::Point3;
#[cfg(feature = "dim3")]
use kiss3d::prelude::InstanceData;
#[cfg(feature = "dim3")]
use kiss3d::scene::SceneNode;
use kiss3d::window::Window;
use rapier::math::DIM;
use rapier::parry::shape::ShapeType;
use std::collections::HashMap;
use std::ops::MulAssign;
#[cfg(feature = "dim2")]
use {
    kiss3d::{
        prelude::{PlanarInstanceData, PlanarSceneNode},
        resource::PlanarMesh,
    },
    nalgebra::{Point2, UnitComplex, Vector2},
    rapier::parry::bounding_volume::Aabb,
    std::cell::RefCell,
    std::rc::Rc,
};

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
    let fixed_color = Point3::new(0.6, 0.6, 0.6);

    let mut instances = HashMap::new();
    let mut singletons = vec![];

    for (i, (_, co)) in phys.colliders.iter().enumerate() {
        let shape = co.shape();
        let is_fixed = co.parent().map(|h| phys.bodies[h].is_fixed()) != Some(false);
        let color = if is_fixed {
            fixed_color
        } else {
            let coeff = (1.0 - 0.15 * (i % 5) as f32) / 255.0;
            match shape.shape_type() {
                ShapeType::Ball => Point3::new(55.0, 126.0, 184.0) * coeff,
                ShapeType::Cuboid => Point3::new(55.0, 126.0, 34.0) * coeff,
                #[cfg(feature = "dim3")]
                ShapeType::Cylinder => Point3::new(140.0, 86.0, 75.0) * coeff,
                #[cfg(feature = "dim3")]
                ShapeType::Cone => Point3::new(255.0, 217.0, 47.0) * coeff,
                ShapeType::Capsule => Point3::new(204.0, 121.0, 167.0) * coeff,
                #[cfg(feature = "dim3")]
                ShapeType::ConvexPolyhedron => Point3::new(228.0, 26.0, 28.0) * coeff,
                _ => Point3::new(255.0, 127.0, 0.0) * coeff,
            }
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
                        window.add_render_mesh(lowres_sphere, nalgebra::Vector3::repeat(1.0))
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
                    color: [color.x, color.y, color.z, 1.0],
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
            #[cfg(feature = "dim3")]
            ShapeType::Cylinder => {
                let instanced_node = instances.entry(ShapeType::Cylinder).or_insert_with(|| {
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
            #[cfg(feature = "dim3")]
            ShapeType::Cone => {
                let instanced_node = instances.entry(ShapeType::Cone).or_insert_with(|| {
                    let node = window.add_cone(1.0, 1.0);
                    InstancedNode {
                        node,
                        entries: vec![],
                        data: vec![],
                    }
                });
                let c = shape.as_cone().unwrap();
                instanced_node.entries.push(InstancedNodeEntry {
                    index: i,
                    color: [color.x, color.y, color.z, 1.0],
                    scale: [c.radius, c.half_height * 2.0, c.radius],
                })
            }
            ShapeType::Capsule => {
                let instanced_node = instances.entry(ShapeType::Capsule).or_insert_with(|| {
                    #[cfg(feature = "dim2")]
                    let node = window.add_planar_capsule(0.5, 1.0);
                    #[cfg(feature = "dim3")]
                    let node = window.add_capsule(0.5, 1.0);
                    InstancedNode {
                        node,
                        entries: vec![],
                        data: vec![],
                    }
                });
                let c = shape.as_capsule().unwrap();
                instanced_node.entries.push(InstancedNodeEntry {
                    index: i,
                    color: [color.x, color.y, color.z, 1.0],
                    #[cfg(feature = "dim2")]
                    scale: [c.radius * 2.0, c.segment.length()],
                    #[cfg(feature = "dim3")]
                    scale: [c.radius * 2.0, c.segment.length(), c.radius * 2.0],
                })
            }
            #[cfg(feature = "dim2")]
            ShapeType::ConvexPolygon => {
                let poly = shape.as_convex_polygon().unwrap();
                let node = window
                    .add_convex_polygon(poly.points().to_vec(), nalgebra::Vector2::repeat(1.0));

                let mut singleton = InstancedNode {
                    node,
                    entries: vec![],
                    data: vec![],
                };
                singleton.entries.push(InstancedNodeEntry {
                    index: i,
                    color: [color.x, color.y, color.z, 1.0],
                    scale: [1.0, 1.0],
                });
                singletons.push(singleton);
            }
            #[cfg(feature = "dim3")]
            ShapeType::ConvexPolyhedron => {
                use kiss3d::procedural::RenderMesh;

                let poly = shape.as_convex_polyhedron().unwrap();
                let (vtx, idx) = poly.to_trimesh();
                let trimesh = rapier::parry::shape::TriMesh::new(vtx, idx).unwrap();
                let mut render = RenderMesh::from(trimesh);
                // Use face normals as vertex normals for flat shading.
                render.replicate_vertices();
                render.recompute_normals();
                let node = window.add_render_mesh(render, nalgebra::Vector3::repeat(1.0));
                let mut singleton = InstancedNode {
                    node,
                    entries: vec![],
                    data: vec![],
                };
                singleton.entries.push(InstancedNodeEntry {
                    index: i,
                    color: [color.x, color.y, color.z, 1.0],
                    scale: [1.0, 1.0, 1.0],
                });
                singletons.push(singleton);
            }
            #[cfg(feature = "dim3")]
            ShapeType::TriMesh => {
                use kiss3d::procedural::RenderMesh;

                let trimesh = shape.as_trimesh().unwrap();
                let mut render = RenderMesh::from(trimesh.clone());
                // Use face normals as vertex normals for flat shading.
                render.replicate_vertices();
                render.recompute_normals();
                let node = window.add_render_mesh(render, nalgebra::Vector3::repeat(1.0));
                let mut singleton = InstancedNode {
                    node,
                    entries: vec![],
                    data: vec![],
                };
                singleton.entries.push(InstancedNodeEntry {
                    index: i,
                    color: [color.x, color.y, color.z, 1.0],
                    scale: [1.0; DIM],
                });
                singletons.push(singleton);
            }
            #[cfg(feature = "dim2")]
            ShapeType::Polyline => {
                let polyline = shape.as_polyline().unwrap();
                let mut vtx = vec![];
                let mut idx = vec![];
                for segment in polyline.segments() {
                    let thickness = 0.2;
                    let center = nalgebra::center(&segment.a, &segment.b);
                    let length = nalgebra::distance(&segment.a, &segment.b);
                    let rot =
                        UnitComplex::rotation_between(&Vector2::y(), &segment.scaled_direction());
                    let aabb = Aabb::from_half_extents(
                        Point2::origin(),
                        Vector2::new(thickness, length / 2.0),
                    );
                    let (mut seg_vtx, mut seg_idx) = aabb.to_trimesh();
                    seg_vtx.iter_mut().for_each(|pt| {
                        *pt = rot * *pt + center.coords;
                    });
                    seg_idx.iter_mut().for_each(|ii| {
                        ii[0] += vtx.len() as u32;
                        ii[1] += vtx.len() as u32;
                        ii[2] += vtx.len() as u32;
                    });
                    vtx.extend_from_slice(&seg_vtx);
                    idx.extend(seg_idx.iter().map(|ii| Point3::from(ii.map(|i| i as u16))));
                }
                let node = window.add_planar_mesh(
                    Rc::new(RefCell::new(PlanarMesh::new(vtx, idx, None, false))),
                    Vector2::repeat(1.0),
                );
                let mut singleton = InstancedNode {
                    node,
                    entries: vec![],
                    data: vec![],
                };
                singleton.entries.push(InstancedNodeEntry {
                    index: i,
                    color: [color.x, color.y, color.z, 1.0],
                    scale: [1.0; DIM],
                });
                singletons.push(singleton);
            }
            _ => todo!(),
        }
    }

    RenderContext {
        instances: (instances.into_values().chain(singletons.into_iter())).collect(),
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
