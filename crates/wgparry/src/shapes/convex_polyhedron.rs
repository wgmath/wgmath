//! Convex polyhedron shape.

use crate::bounding_volumes::WgAabb;
use crate::queries::{WgPolygonalFeature, WgProjection, WgRay};
use crate::shapes::WgVtxIdx;
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
        WgVtxIdx
    ),
    src = "convex_polyhedron.wgsl",
    src_fn = "substitute_aliases",
    shader_defs = "dim_shader_defs"
)]
/// GPU shader for the convex polyhedron shape.
///
/// The convex polyhedron is defined by a vertex buffer and a triangle index buffer.
/// Both convex polyhedrons and triangle meshes share the same GPU buffer for storing their
/// vertices and indices.
pub struct WgConvexPolyhedron;
