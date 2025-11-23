//! Triangle mesh shape.

use crate::bounding_volumes::WgAabb;
use crate::queries::{WgPolygonalFeature, WgProjection, WgRay};
use crate::shapes::{WgConvexPolyhedron, WgTriangle, WgVtxIdx};
use crate::{dim_shader_defs, substitute_aliases};
use wgcore::Shader;
use wgebra::{WgSim2, WgSim3};

#[derive(Shader)]
#[shader(
    derive(
        WgSim3,
        WgSim2,
        WgRay,
        WgProjection,
        WgPolygonalFeature,
        WgAabb,
        WgTriangle,
        WgConvexPolyhedron,
        WgVtxIdx,
    ),
    src = "trimesh.wgsl",
    src_fn = "substitute_aliases",
    shader_defs = "dim_shader_defs"
)]
/// GPU shader for the trimesh shape.
///
/// The trimesh is defined by a BVH, vertex buffer and a triangle index buffer.
/// The BVH is stored as part of the vertex and index buffer, before the actual vertices and indices.
/// It stores two vectors and one index per AABB.
pub struct WgTriMesh;
