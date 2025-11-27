//! Unified shape representation for GPU-accelerated collision detection.
//!
//! This module provides [`GpuShape`], a compact, GPU-friendly representation that can
//! encode any supported geometric primitive. It uses a tagged union approach where
//! shape data is packed into two `Vector4<f32>` values, with the shape type encoded
//! in the tag field.
//!
//! # Shape Encoding
//!
//! Different shapes pack their parameters differently:
//! - **Ball**: `a = [radius, _, _, tag]`
//! - **Cuboid**: `a = [hx, hy, hz, tag]` (half-extents)
//! - **Capsule**: `a = [ax, ay, az, tag]`, `b = [bx, by, bz, radius]` (segment endpoints + radius)
//! - **Cone**: `a = [half_height, radius, _, tag]`
//! - **Cylinder**: `a = [half_height, radius, _, tag]`
//! - **Polyline/TriMesh**: `a = [range_start, range_end, _, tag]` (vertex buffer indices)
//!
//! The tag is stored as the `w` component of the first vector using bit-casting to preserve
//! the `f32` representation while encoding a `u32` shape type identifier.

use crate::bounding_volumes::WgAabb;
use crate::queries::{WgPolygonalFeature, WgProjection, WgRay};
use crate::shapes::{
    WgBall, WgCapsule, WgConvexPolyhedron, WgCuboid, WgPolyline, WgTriMesh, WgTriangle,
};
use crate::{dim_shader_defs, substitute_aliases};
use na::{vector, Vector4};
use parry::bounding_volume::Aabb;
use parry::shape::{Ball, Cuboid, Shape, ShapeType, TypedShape};
use wgcore::{test_shader_compilation, Shader};
use wgebra::{WgSim2, WgSim3};

use crate::math::{Point, Vector, DIM};
#[cfg(feature = "dim3")]
use crate::shapes::cone::WgCone;
#[cfg(feature = "dim3")]
use crate::shapes::cylinder::WgCylinder;

/// Shape type identifiers for GPU representation.
///
/// These numeric values are encoded in the tag field of [`GpuShape`] and must match
/// the corresponding values in `shape.wgsl`. The values are bit-cast as `f32` for
/// storage in GPU buffers.
pub enum GpuShapeType {
    /// Ball/sphere shape (type ID = 0)
    Ball = 0,
    /// Cuboid/box shape (type ID = 1)
    Cuboid = 1,
    /// Capsule shape (type ID = 2)
    Capsule = 2,
    #[cfg(feature = "dim3")]
    /// Cone shape, 3D only (type ID = 3)
    Cone = 3,
    #[cfg(feature = "dim3")]
    /// Cylinder shape, 3D only (type ID = 4)
    Cylinder = 4,
    /// Polyline - sequence of connected line segments (type ID = 5)
    // TODO: not sure we want to keep the Polyline in the shape type.
    Polyline = 5,
    /// Triangle mesh (type ID = 6)
    TriMesh = 6,
    /// A convex polygon or polyhedron (type ID = 7)
    ConvexPoly = 7,
}

/// Auxiliary buffers for complex shape types like polylines and triangle meshes.
///
/// Some shapes (polylines and triangle meshes) reference external vertex data
/// rather than storing all data inline. This struct holds those vertex buffers.
#[derive(Default, Clone, Debug)]
pub struct ShapeBuffers {
    /// Vertex buffer for polylines and triangle meshes.
    ///
    /// Polyline and TriMesh shapes reference ranges within this buffer.
    /// The shape stores the start and end indices of its vertices in this buffer.
    pub vertices: Vec<Point<f32>>,
    /// Index buffers for polylines, triangle meshes, and convex polyhedrons.
    pub indices: Vec<u32>,
}

/// GPU-compatible shape representation using a tagged union encoded in two `Vector4<f32>` values.
///
/// This struct provides a compact, cache-friendly representation of various geometric
/// primitives suitable for GPU buffer storage. The shape type is encoded in the `w`
/// component of the first vector as a bit-cast `u32` value.
///
/// # Memory Layout
///
/// The struct is `#[repr(C)]` with `bytemuck::Pod` for direct GPU buffer uploads.
/// Total size: 32 bytes (2 × `Vector4<f32>`).
///
/// # Supported Operations
///
/// - Construct shapes from primitive parameters (e.g., [`ball`](Self::ball), [`cuboid`](Self::cuboid))
/// - Convert from parry shape types via [`from_parry`](Self::from_parry)
/// - Query shape type with [`shape_type`](Self::shape_type)
/// - Extract typed shapes with [`to_ball`](Self::to_ball), [`to_cuboid`](Self::to_cuboid), etc.
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct GpuShape {
    a: Vector4<f32>,
    b: Vector4<f32>,
    // Needed for triangles. While we don’t construct explicitly triangle shapes on the rust side,
    // this is useful on the GPU so that collision-detection involving triangle meshes can then
    // call the pfm-pfm collision-detection (which relies on dynamic dispatch) by storing the
    // triangle as a Shape.
    c: Vector4<f32>,
}

impl GpuShape {
    /// Creates a ball (sphere/circle) shape.
    ///
    /// # Parameters
    ///
    /// - `radius`: The radius of the ball
    ///
    /// # Returns
    ///
    /// A [`GpuShape`] encoding a ball with the specified radius
    pub fn ball(radius: f32) -> Self {
        let tag = f32::from_bits(GpuShapeType::Ball as u32);
        Self {
            a: vector![radius, 0.0, 0.0, tag],
            b: vector![0.0, 0.0, 0.0, 0.0],
            c: vector![0.0, 0.0, 0.0, 0.0],
        }
    }

    /// Creates a cuboid (box/rectangle) shape.
    ///
    /// # Parameters
    ///
    /// - `half_extents`: Half-widths along each axis (vec2 for 2D, vec3 for 3D)
    ///
    /// # Returns
    ///
    /// A [`GpuShape`] encoding a cuboid with the specified dimensions
    pub fn cuboid(half_extents: Vector<f32>) -> Self {
        let tag = f32::from_bits(GpuShapeType::Cuboid as u32);
        Self {
            #[cfg(feature = "dim2")]
            a: vector![half_extents.x, half_extents.y, 0.0, tag],
            #[cfg(feature = "dim3")]
            a: vector![half_extents.x, half_extents.y, half_extents.z, tag],
            b: vector![0.0, 0.0, 0.0, 0.0],
            c: vector![0.0, 0.0, 0.0, 0.0],
        }
    }

    /// Creates a capsule shape.
    ///
    /// # Parameters
    ///
    /// - `a`: First endpoint of the capsule's central segment
    /// - `b`: Second endpoint of the capsule's central segment
    /// - `radius`: Radius of the capsule (distance from segment to surface)
    ///
    /// # Returns
    ///
    /// A [`GpuShape`] encoding a capsule with the specified parameters
    pub fn capsule(a: Point<f32>, b: Point<f32>, radius: f32) -> Self {
        let tag = f32::from_bits(GpuShapeType::Capsule as u32);
        #[cfg(feature = "dim2")]
        return Self {
            a: vector![a.x, a.y, 0.0, tag],
            b: vector![b.x, b.y, 0.0, radius],
            c: vector![0.0, 0.0, 0.0, 0.0],
        };
        #[cfg(feature = "dim3")]
        return Self {
            a: vector![a.x, a.y, a.z, tag],
            b: vector![b.x, b.y, b.z, radius],
            c: vector![0.0, 0.0, 0.0, 0.0],
        };
    }

    /// Creates a polyline shape from a range of vertices.
    ///
    /// A polyline is a connected sequence of line segments defined by vertices.
    ///
    /// # Parameters
    ///
    /// - `vertex_range`: Start and end indices into the vertex buffer
    pub fn polyline(
        bvh_vtx_root_id: u32,
        bvh_idx_root_id: u32,
        bvh_node_len: u32,
        aabb: Aabb,
    ) -> Self {
        let tag = f32::from_bits(GpuShapeType::Polyline as u32);
        let a0 = f32::from_bits(bvh_vtx_root_id);
        let a1 = f32::from_bits(bvh_idx_root_id);
        let a2 = f32::from_bits(bvh_node_len);
        #[cfg(feature = "dim2")]
        return Self {
            a: vector![a0, a1, a2, tag],
            b: vector![aabb.mins.x, aabb.mins.y, 0.0, 0.0],
            c: vector![aabb.maxs.x, aabb.maxs.y, 0.0, 0.0],
        };
        #[cfg(feature = "dim3")]
        return Self {
            a: vector![a0, a1, a2, tag],
            b: aabb.mins.coords.push(0.0),
            c: aabb.maxs.coords.push(0.0),
        };
    }

    /// Creates a triangle mesh shape from a range of vertices.
    ///
    /// A trimesh is a collection of triangles sharing vertices.
    pub fn trimesh(
        bvh_vtx_root_id: u32,
        bvh_idx_root_id: u32,
        bvh_node_len: u32,
        aabb: Aabb,
    ) -> Self {
        let tag = f32::from_bits(GpuShapeType::TriMesh as u32);
        let a0 = f32::from_bits(bvh_vtx_root_id);
        let a1 = f32::from_bits(bvh_idx_root_id);
        let a2 = f32::from_bits(bvh_node_len);
        #[cfg(feature = "dim2")]
        return Self {
            a: vector![a0, a1, a2, tag],
            b: vector![aabb.mins.x, aabb.mins.y, 0.0, 0.0],
            c: vector![aabb.maxs.x, aabb.maxs.y, 0.0, 0.0],
        };
        #[cfg(feature = "dim3")]
        return Self {
            a: vector![a0, a1, a2, tag],
            b: aabb.mins.coords.push(0.0),
            c: aabb.maxs.coords.push(0.0),
        };
    }

    /// Creates a convex polyhedron definition from its vertex and index buffer anges.
    pub fn convex_poly(
        first_vtx_id: u32,
        end_vtx_id: u32,
        first_face_id: u32,
        end_face_id: u32,
    ) -> Self {
        let tag = f32::from_bits(GpuShapeType::ConvexPoly as u32);
        let a0 = f32::from_bits(first_vtx_id);
        let a1 = f32::from_bits(end_vtx_id);
        let b0 = f32::from_bits(first_face_id);
        let b1 = f32::from_bits(end_face_id);
        Self {
            a: vector![a0, a1, 0.0, tag],
            b: vector![b0, b1, 0.0, 0.0],
            c: vector![0.0, 0.0, 0.0, 0.0],
        }
    }

    /// Creates a cone shape (3D only).
    ///
    /// # Parameters
    ///
    /// - `half_height`: Half the height of the cone (from base to apex)
    /// - `radius`: Radius of the cone's base
    #[cfg(feature = "dim3")]
    pub fn cone(half_height: f32, radius: f32) -> Self {
        let tag = f32::from_bits(GpuShapeType::Cone as u32);
        Self {
            a: vector![half_height, radius, 0.0, tag],
            b: vector![0.0, 0.0, 0.0, 0.0],
            c: vector![0.0, 0.0, 0.0, 0.0],
        }
    }

    /// Creates a cylinder shape (3D only).
    ///
    /// # Parameters
    ///
    /// - `half_height`: Half the height of the cylinder (distance from center to end caps)
    /// - `radius`: Radius of the cylinder
    #[cfg(feature = "dim3")]
    pub fn cylinder(half_height: f32, radius: f32) -> Self {
        let tag = f32::from_bits(GpuShapeType::Cylinder as u32);
        Self {
            a: vector![half_height, radius, 0.0, tag],
            b: vector![0.0, 0.0, 0.0, 0.0],
            c: vector![0.0, 0.0, 0.0, 0.0],
        }
    }

    /// Converts a parry shape to a [`GpuShape`].
    ///
    /// This method handles conversion from parry's CPU-side shape types to the GPU-compatible
    /// representation. For complex shapes like polylines and triangle meshes, vertex data is
    /// appended to the provided buffers.
    ///
    /// # Parameters
    ///
    /// - `shape`: The parry shape to convert
    /// - `buffers`: Vertex buffers for storing polyline/mesh vertex data
    ///
    /// # Returns
    ///
    /// - `Some(GpuShape)` if the shape type is supported
    /// - `None` if the shape type is not yet supported on GPU
    ///
    /// # Supported Shape Types
    ///
    /// - Ball, Cuboid, Capsule (primitives)
    /// - Cone, Cylinder (3D only)
    /// - Polyline, TriMesh, HeightField (complex shapes stored as vertex ranges)
    pub fn from_parry(shape: &(impl Shape + ?Sized), buffers: &mut ShapeBuffers) -> Option<Self> {
        match shape.as_typed_shape() {
            TypedShape::Ball(shape) => Some(Self::ball(shape.radius)),
            TypedShape::Cuboid(shape) => Some(Self::cuboid(shape.half_extents)),
            TypedShape::Capsule(shape) => Some(Self::capsule(
                shape.segment.a,
                shape.segment.b,
                shape.radius,
            )),
            TypedShape::Polyline(shape) => {
                let bvh_vtx_root_id = buffers.vertices.len();
                let bvh_idx_root_id = buffers.indices.len();
                // Append the BVH data to the vertex/index buffers.
                // TODO: we are constructing a BVH using the `bvh` crate.
                //       While the Polyline shape technically already has a BVH, parry’s BVH
                //       doesn’t provide explicit access to the BVH topology. So, for now,
                //       let’s just build a new BVH that exposes its internal.
                struct BvhObject {
                    aabb: bvh::aabb::Aabb<f32, DIM>,
                    node_index: usize,
                }

                impl bvh::aabb::Bounded<f32, DIM> for BvhObject {
                    fn aabb(&self) -> bvh::aabb::Aabb<f32, DIM> {
                        self.aabb
                    }
                }

                impl bvh::bounding_hierarchy::BHShape<f32, DIM> for BvhObject {
                    fn set_bh_node_index(&mut self, index: usize) {
                        self.node_index = index;
                    }

                    fn bh_node_index(&self) -> usize {
                        self.node_index
                    }
                }

                let mut objects: Vec<_> = shape
                    .segments()
                    .map(|tri| {
                        let aabb = tri.local_aabb();
                        BvhObject {
                            aabb: bvh::aabb::Aabb::with_bounds(aabb.mins, aabb.maxs),
                            node_index: 0,
                        }
                    })
                    .collect();

                let bvh = bvh::bvh::Bvh::build(&mut objects);
                let flat_bvh = bvh.flatten();
                buffers
                    .vertices
                    .extend(flat_bvh.iter().flat_map(|n| [n.aabb.min, n.aabb.max]));
                let bvh_node_len = flat_bvh.len();
                buffers.indices.extend(
                    flat_bvh
                        .iter()
                        .flat_map(|n| [n.entry_index, n.exit_index, n.shape_index]),
                );

                // Append the actual mesh vertex/index buffers.
                buffers.vertices.extend_from_slice(shape.vertices());
                buffers
                    .indices
                    .extend(shape.indices().iter().flat_map(|seg| seg.iter().copied()));
                Some(Self::polyline(
                    bvh_vtx_root_id as u32,
                    bvh_idx_root_id as u32,
                    bvh_node_len as u32,
                    shape.local_aabb(),
                ))
            }
            TypedShape::TriMesh(shape) => {
                let bvh_vtx_root_id = buffers.vertices.len();
                let bvh_idx_root_id = buffers.indices.len();
                // Append the BVH data to the vertex/index buffers.
                // TODO: we are constructing a BVH using the `bvh` crate.
                //       While the TriMesh shape technically already has a BVH, parry’s BVH
                //       doesn’t provide explicit access to the BVH topology. So, for now,
                //       let’s just build a new BVH that exposes its internal.
                struct BvhObject {
                    aabb: bvh::aabb::Aabb<f32, DIM>,
                    node_index: usize,
                }

                impl bvh::aabb::Bounded<f32, DIM> for BvhObject {
                    fn aabb(&self) -> bvh::aabb::Aabb<f32, DIM> {
                        self.aabb
                    }
                }

                impl bvh::bounding_hierarchy::BHShape<f32, DIM> for BvhObject {
                    fn set_bh_node_index(&mut self, index: usize) {
                        self.node_index = index;
                    }

                    fn bh_node_index(&self) -> usize {
                        self.node_index
                    }
                }

                let mut objects: Vec<_> = shape
                    .triangles()
                    .map(|tri| {
                        let aabb = tri.local_aabb();
                        BvhObject {
                            aabb: bvh::aabb::Aabb::with_bounds(aabb.mins, aabb.maxs),
                            node_index: 0,
                        }
                    })
                    .collect();

                let bvh = bvh::bvh::Bvh::build(&mut objects);
                let flat_bvh = bvh.flatten();
                buffers
                    .vertices
                    .extend(flat_bvh.iter().flat_map(|n| [n.aabb.min, n.aabb.max]));
                let bvh_node_len = flat_bvh.len();
                buffers.indices.extend(
                    flat_bvh
                        .iter()
                        .flat_map(|n| [n.entry_index, n.exit_index, n.shape_index]),
                );

                // Append the actual mesh vertex/index buffers.
                buffers.vertices.extend_from_slice(shape.vertices());
                buffers
                    .indices
                    .extend(shape.indices().iter().flat_map(|tri| tri.iter().copied()));
                Some(Self::trimesh(
                    bvh_vtx_root_id as u32,
                    bvh_idx_root_id as u32,
                    bvh_node_len as u32,
                    shape.local_aabb(),
                ))
            }
            #[cfg(feature = "dim2")]
            TypedShape::ConvexPolygon(poly) => {
                let first_vtx_id = buffers.vertices.len() as u32;
                buffers.vertices.extend_from_slice(poly.points());
                let end_vtx_id = buffers.vertices.len() as u32;
                // NOTE: face ids are not relevant for 2D convex polyhedron since it dosn’t have
                //       an index buffer.
                Some(Self::convex_poly(first_vtx_id, end_vtx_id, 0, 0))
            }
            #[cfg(feature = "dim3")]
            TypedShape::ConvexPolyhedron(poly) => {
                let first_vtx_id = buffers.vertices.len();
                let first_face_id = buffers.indices.len();
                let all_idx = poly.vertices_adj_to_face();

                buffers.vertices.extend_from_slice(poly.points());
                for face in poly.faces() {
                    let id = face.first_vertex_or_edge as usize;

                    if face.num_vertices_or_edges < 3 {
                        println!("found convex poly with degenerate faces?")
                    } else {
                        buffers.indices.push(all_idx[id]);
                        buffers.indices.push(all_idx[id + 1]);
                        buffers.indices.push(all_idx[id + 2]);
                    }
                }

                let end_vtx_id = buffers.vertices.len();
                let end_face_id = buffers.indices.len();
                Some(Self::convex_poly(
                    first_vtx_id as u32,
                    end_vtx_id as u32,
                    first_face_id as u32,
                    end_face_id as u32,
                ))
            }
            // HACK: we currently emulate heightfields as trimeshes or polylines
            #[cfg(feature = "dim2")]
            TypedShape::HeightField(_shape) => {
                todo!()
            }
            #[cfg(feature = "dim3")]
            TypedShape::HeightField(_shape) => {
                todo!()
            }
            #[cfg(feature = "dim3")]
            TypedShape::Cone(shape) => Some(Self::cone(shape.half_height, shape.radius)),
            #[cfg(feature = "dim3")]
            TypedShape::Cylinder(shape) => Some(Self::cylinder(shape.half_height, shape.radius)),
            _ => None,
        }
    }

    /// Returns the shape type of this [`GpuShape`].
    ///
    /// Extracts the shape type tag from the `w` component of the first vector.
    ///
    /// # Returns
    ///
    /// The [`ShapeType`] corresponding to the encoded shape
    pub fn shape_type(&self) -> ShapeType {
        let tag = self.a.w.to_bits();

        match tag {
            0 => ShapeType::Ball,
            1 => ShapeType::Cuboid,
            2 => ShapeType::Capsule,
            #[cfg(feature = "dim3")]
            3 => ShapeType::Cone,
            #[cfg(feature = "dim3")]
            4 => ShapeType::Cylinder,
            5 => ShapeType::Polyline,
            6 => ShapeType::TriMesh,
            _ => panic!("Unknown shape type: {}", tag),
        }
    }

    /// Extracts a [`Ball`] if this shape is a ball.
    ///
    /// # Returns
    ///
    /// - `Some(Ball)` if the shape type is Ball
    /// - `None` otherwise
    pub fn to_ball(&self) -> Option<Ball> {
        (self.shape_type() == ShapeType::Ball).then_some(Ball::new(self.a.x))
    }

    /// Extracts a [`Cuboid`] if this shape is a cuboid.
    ///
    /// # Returns
    ///
    /// - `Some(Cuboid)` if the shape type is Cuboid
    /// - `None` otherwise
    pub fn to_cuboid(&self) -> Option<Cuboid> {
        #[cfg(feature = "dim2")]
        return (self.shape_type() == ShapeType::Cuboid).then_some(Cuboid::new(self.a.xy()));
        #[cfg(feature = "dim3")]
        return (self.shape_type() == ShapeType::Cuboid).then_some(Cuboid::new(self.a.xyz()));
    }

    /// Returns the vertex buffer range for a polyline shape.
    ///
    /// # Returns
    ///
    /// A `[start, end)` index range into the vertex buffer
    ///
    /// # Panics
    ///
    /// Panics if this shape is not a polyline
    pub fn polyline_rngs(&self) -> [u32; 2] {
        assert!(self.shape_type() == ShapeType::Polyline);
        [self.a.x.to_bits(), self.a.y.to_bits()]
    }

    /// Returns the vertex buffer range for a triangle mesh shape.
    ///
    /// # Returns
    ///
    /// A `[start, end)` index range into the vertex buffer
    ///
    /// # Panics
    ///
    /// Panics if this shape is not a triangle mesh
    pub fn trimesh_rngs(&self) -> [u32; 2] {
        assert!(self.shape_type() == ShapeType::TriMesh);
        [self.a.x.to_bits(), self.a.y.to_bits()]
    }
}

#[cfg(feature = "dim2")]
#[derive(Shader)]
#[shader(src = "shape_fake_cone.wgsl")]
/// Fake cone shader for 2D builds to satisfy shader composition dependencies.
///
/// In 2D builds, cones don't exist but are still referenced by the unified shape shader.
/// This provides a stub implementation to prevent compilation errors.
struct WgCone;

#[cfg(feature = "dim2")]
#[derive(Shader)]
#[shader(src = "shape_fake_cylinder.wgsl")]
/// Fake cylinder shader for 2D builds to satisfy shader composition dependencies.
///
/// In 2D builds, cylinders don't exist but are still referenced by the unified shape shader.
/// This provides a stub implementation to prevent compilation errors.
struct WgCylinder;

#[derive(Shader)]
#[shader(
    derive(
        WgSim3,
        WgSim2,
        WgRay,
        WgProjection,
        WgBall,
        WgCapsule,
        WgCone,
        WgCuboid,
        WgCylinder,
        WgPolygonalFeature,
        WgConvexPolyhedron,
        WgTriangle,
        WgTriMesh,
        WgPolyline,
        WgAabb,
    ),
    src = "shape.wgsl",
    src_fn = "substitute_aliases",
    shader_defs = "dim_shader_defs"
)]
/// Unified GPU shader for all shape types.
///
/// This shader provides a unified interface for geometric queries on any supported
/// shape type. It uses dynamic dispatch based on the shape's type tag to call the
/// appropriate specialized shader implementation.
pub struct WgShape;

test_shader_compilation!(WgShape, wgcore, crate::dim_shader_defs());
