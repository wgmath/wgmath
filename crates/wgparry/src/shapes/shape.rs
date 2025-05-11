//! A shape enum.

use crate::queries::{WgProjection, WgRay};
use crate::shapes::{WgBall, WgCapsule, WgCuboid};
use crate::{dim_shader_defs, substitute_aliases};
use na::{vector, Vector4};
use parry::shape::{Shape, ShapeType, TypedShape};
use wgcore::{test_shader_compilation, Shader};
use wgebra::{WgSim2, WgSim3};

use crate::math::{Point, Vector};
#[cfg(feature = "dim3")]
use crate::shapes::cone::WgCone;
#[cfg(feature = "dim3")]
use crate::shapes::cylinder::WgCylinder;

// NOTE: this must match the type values in shape.wgsl
pub enum GpuShapeType {
    Ball = 0,
    Cuboid = 1,
    Capsule = 2,
    #[cfg(feature = "dim3")]
    Cone = 3,
    #[cfg(feature = "dim3")]
    Cylinder = 4,
    // TODO: not sure we want to keep the Polyline in the shape type.
    Polyline = 5,
    TriMesh = 6,
}

#[derive(Default, Clone, Debug)]
pub struct ShapeBuffers {
    pub vertices: Vec<Point<f32>>,
    // NOTE: a bit weird we don’t have any index buffer here but
    //       we don’t need it yet (wgsparkl has its own indexing method).
}

#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct GpuShape {
    a: Vector4<f32>,
    b: Vector4<f32>,
}

impl GpuShape {
    pub fn ball(radius: f32) -> Self {
        let tag = f32::from_bits(GpuShapeType::Ball as u32);
        Self {
            a: vector![radius, 0.0, 0.0, tag],
            b: vector![0.0, 0.0, 0.0, 0.0],
        }
    }

    pub fn cuboid(half_extents: Vector<f32>) -> Self {
        let tag = f32::from_bits(GpuShapeType::Cuboid as u32);
        Self {
            #[cfg(feature = "dim2")]
            a: vector![half_extents.x, half_extents.y, 0.0, tag],
            #[cfg(feature = "dim3")]
            a: vector![half_extents.x, half_extents.y, half_extents.z, tag],
            b: vector![0.0, 0.0, 0.0, 0.0],
        }
    }

    pub fn capsule(a: Point<f32>, b: Point<f32>, radius: f32) -> Self {
        let tag = f32::from_bits(GpuShapeType::Capsule as u32);
        #[cfg(feature = "dim2")]
        return Self {
            a: vector![a.x, a.y, 0.0, tag],
            b: vector![b.x, b.y, 0.0, radius],
        };
        #[cfg(feature = "dim3")]
        return Self {
            a: vector![a.x, a.y, a.z, tag],
            b: vector![b.x, b.y, b.z, radius],
        };
    }

    pub fn polyline(vertex_range: [u32; 2]) -> Self {
        let tag = f32::from_bits(GpuShapeType::Polyline as u32);
        let rng0 = f32::from_bits(vertex_range[0]);
        let rng1 = f32::from_bits(vertex_range[1]);
        Self {
            a: vector![rng0, rng1, 0.0, tag],
            b: vector![0.0, 0.0, 0.0, 0.0],
        }
    }

    pub fn trimesh(vertex_range: [u32; 2]) -> Self {
        let tag = f32::from_bits(GpuShapeType::TriMesh as u32);
        let rng0 = f32::from_bits(vertex_range[0]);
        let rng1 = f32::from_bits(vertex_range[1]);
        Self {
            a: vector![rng0, rng1, 0.0, tag],
            b: vector![0.0, 0.0, 0.0, 0.0],
        }
    }

    #[cfg(feature = "dim3")]
    pub fn cone(half_height: f32, radius: f32) -> Self {
        let tag = f32::from_bits(GpuShapeType::Cone as u32);
        Self {
            a: vector![half_height, radius, 0.0, tag],
            b: vector![0.0, 0.0, 0.0, 0.0],
        }
    }

    #[cfg(feature = "dim3")]
    pub fn cylinder(half_height: f32, radius: f32) -> Self {
        let tag = f32::from_bits(GpuShapeType::Cylinder as u32);
        Self {
            a: vector![half_height, radius, 0.0, tag],
            b: vector![0.0, 0.0, 0.0, 0.0],
        }
    }

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
                let base_id = buffers.vertices.len();
                buffers.vertices.extend_from_slice(shape.vertices());
                Some(Self::polyline([
                    base_id as u32,
                    buffers.vertices.len() as u32,
                ]))
            }
            TypedShape::TriMesh(shape) => {
                let base_id = buffers.vertices.len();
                buffers.vertices.extend_from_slice(shape.vertices());
                Some(Self::trimesh([
                    base_id as u32,
                    buffers.vertices.len() as u32,
                ]))
            }
            // HACK: we currently emulate heightfields as trimeshes or polylines
            #[cfg(feature = "dim2")]
            TypedShape::HeightField(shape) => {
                let base_id = buffers.vertices.len();
                let (vtx, _) = shape.to_polyline();
                buffers.vertices.extend_from_slice(&vtx);
                Some(Self::polyline([
                    base_id as u32,
                    buffers.vertices.len() as u32,
                ]))
            }
            #[cfg(feature = "dim3")]
            TypedShape::HeightField(shape) => {
                let base_id = buffers.vertices.len();
                let (vtx, _) = shape.to_trimesh();
                buffers.vertices.extend_from_slice(&vtx);
                Some(Self::trimesh([
                    base_id as u32,
                    buffers.vertices.len() as u32,
                ]))
            }
            #[cfg(feature = "dim3")]
            TypedShape::Cone(shape) => Some(Self::cone(shape.half_height, shape.radius)),
            #[cfg(feature = "dim3")]
            TypedShape::Cylinder(shape) => Some(Self::cylinder(shape.half_height, shape.radius)),
            _ => None,
        }
    }

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

    pub fn polyline_rngs(&self) -> [u32; 2] {
        assert!(self.shape_type() == ShapeType::Polyline);
        [self.a.x.to_bits(), self.a.y.to_bits()]
    }

    pub fn trimesh_rngs(&self) -> [u32; 2] {
        assert!(self.shape_type() == ShapeType::TriMesh);
        [self.a.x.to_bits(), self.a.y.to_bits()]
    }
}

#[cfg(feature = "dim2")]
#[derive(Shader)]
#[shader(src = "shape_fake_cone.wgsl")]
// A fake Cone kernel to work around some naga-oil issues.
struct WgCone;
#[cfg(feature = "dim2")]
#[derive(Shader)]
#[shader(src = "shape_fake_cylinder.wgsl")]
// A fake Cone kernel to work around some naga-oil issues.
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
        WgCylinder
    ),
    src = "shape.wgsl",
    src_fn = "substitute_aliases",
    shader_defs = "dim_shader_defs"
)]
/// Shader defining the shape as well as its ray-casting and point-projection functions.
pub struct WgShape;

test_shader_compilation!(WgShape, wgcore, crate::dim_shader_defs());
