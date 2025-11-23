//! Triangle shape.
//!
//! A triangle is defined by three vertices (A, B, C). Triangles are the fundamental
//! building blocks of triangle meshes and are essential for representing complex
//! geometric surfaces in collision detection.

use crate::bounding_volumes::WgAabb;
use crate::queries::{WgPolygonalFeature, WgProjection, WgRay};
use crate::shapes::WgSegment;
use crate::substitute_aliases;
use wgcore::Shader;
use wgebra::{WgSim2, WgSim3};

#[derive(Shader)]
#[shader(
    derive(
        WgSim3,
        WgSim2,
        WgRay,
        WgProjection,
        WgSegment,
        WgPolygonalFeature,
        WgAabb
    ),
    src = "triangle.wgsl",
    src_fn = "substitute_aliases"
)]
/// GPU shader for the triangle shape.
///
/// This shader provides WGSL implementations for geometric operations on triangles,
/// including ray-casting and point projection.
///
/// A triangle is defined by three vertices (A, B, C) in local space.
pub struct WgTriangle;
