//! Triangle shape.
//!
//! A triangle is defined by three vertices (A, B, C). Triangles are the fundamental
//! building blocks of triangle meshes and are essential for representing complex
//! geometric surfaces in collision detection.

use crate::queries::WgProjection;
use crate::substitute_aliases;
use wgcore::Shader;

#[derive(Shader)]
#[shader(
    derive(WgProjection),
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
